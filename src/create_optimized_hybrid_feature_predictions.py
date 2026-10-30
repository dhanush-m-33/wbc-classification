"""Create hybrid feature outputs from calibrated CNN class likelihoods.

Run from the project root:
    python src/create_optimized_hybrid_feature_predictions.py

The three monocyte-associated feature binaries use a thresholded monocyte
likelihood derived from all CNN feature probabilities. The threshold and
temperature are selected on group-disjoint calibration groups. Evaluation is
reported on separate groups from the original held-out validation fold.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedGroupKFold


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "manifests" / "raabin_train.csv"
DEFAULT_PROCESSED_MANIFEST = (
    PROJECT_ROOT / "data" / "manifests" / "processed_raabin_train.csv"
)
DEFAULT_PREDICTIONS = (
    PROJECT_ROOT / "results" / "raabin_train_feature_predictions.csv"
)
DEFAULT_OUTPUT_PREDICTIONS = (
    PROJECT_ROOT
    / "results"
    / "raabin_train_feature_predictions_hybrid_optimized.csv"
)
DEFAULT_OUTPUT_REPORT = (
    PROJECT_ROOT / "reports" / "raabin_hybrid_optimized_metrics.json"
)
DEFAULT_OUTPUT_SUMMARY = (
    PROJECT_ROOT / "reports" / "raabin_feature_f1_summary.csv"
)
FEATURES = (
    "granules",
    "dense_pink_granules",
    "dense_blue_granules",
    "thin_rim_of_cytoplasm",
    "abundant_cytoplasm",
    "multi_lobed_nucleus_3_5",
    "bilobed_nucleus",
    "kidney_bean_shaped_nucleus",
    "eccentric_and_large_nucleus",
    "small_size_dense_cytoplasm",
)
CLASS_TARGETS = {
    "eosinophil": [1, 1, 0, 0, 0, 0, 1, 0, 0, 0],
    "lymphocyte": [0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
    "monocyte": [0, 0, 0, 0, 1, 0, 0, 1, 1, 0],
    "neutrophil": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
}
CLASSES = ("eosinophil", "lymphocyte", "monocyte", "neutrophil")
MONOCYTE_DERIVED_FEATURES = (
    "abundant_cytoplasm",
    "kidney_bean_shaped_nucleus",
    "eccentric_and_large_nucleus",
)
TEMPERATURES = (0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0)


def _resolve_path(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_matched_rows(manifest_path, processed_manifest_path, predictions):
    manifest = pd.read_csv(manifest_path)
    processed = pd.read_csv(processed_manifest_path)
    required_manifest = {"path", "label", "dataset", "split", "group_id"}
    required_processed = {"raw_path", "path", "label"}
    required_predictions = {"image_filename", "actual_wbc_class"}
    for name, frame, required in (
        ("training manifest", manifest, required_manifest),
        ("processed manifest", processed, required_processed),
        ("prediction file", predictions, required_predictions),
    ):
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"{name} is missing required columns: {missing}")

    manifest["label"] = manifest["label"].astype(str).str.strip().str.lower()
    processed["label"] = processed["label"].astype(str).str.strip().str.lower()
    predictions["actual_wbc_class"] = (
        predictions["actual_wbc_class"].astype(str).str.strip().str.lower()
    )
    if not manifest["dataset"].astype(str).str.lower().eq("raabin").all():
        raise ValueError("The training manifest contains non-RAABIN rows.")
    if not manifest["split"].astype(str).str.lower().eq("train").all():
        raise ValueError("The training manifest must contain only the train split.")

    probability_columns = [f"{feature}_probability" for feature in FEATURES]
    missing_probabilities = sorted(
        set(probability_columns).difference(predictions.columns)
    )
    if missing_probabilities:
        raise ValueError(
            f"Prediction file is missing feature probability columns: "
            f"{missing_probabilities}"
        )

    rows = manifest.merge(
        processed[["raw_path", "path", "label"]].rename(
            columns={"path": "processed_path", "label": "processed_label"}
        ),
        left_on="path",
        right_on="raw_path",
        how="left",
        validate="one_to_one",
    )
    matched = rows["processed_path"].notna()
    label_mismatch = matched & (rows["label"] != rows["processed_label"])
    if label_mismatch.any():
        raise ValueError(
            f"{int(label_mismatch.sum())} processed manifest labels do not match "
            "the training manifest."
        )
    rows = rows.loc[matched].copy()
    rows["image_path"] = rows["processed_path"].map(_resolve_path)
    rows = rows.loc[rows["image_path"].map(Path.is_file)].copy()
    rows["image_filename"] = rows["image_path"].map(lambda path: path.name)
    rows = rows.merge(
        predictions[
            [
                "image_filename",
                "actual_wbc_class",
                *probability_columns,
                *[
                    f"{feature}_binary"
                    for feature in FEATURES
                    if f"{feature}_binary" in predictions.columns
                ],
            ]
        ],
        on="image_filename",
        how="inner",
        validate="one_to_one",
    )
    class_mismatch = rows["label"] != rows["actual_wbc_class"]
    if class_mismatch.any():
        raise ValueError(
            f"{int(class_mismatch.sum())} prediction classes do not match "
            "the training manifest."
        )
    if len(rows) == 0:
        raise ValueError("No processed training images match the predictions.")
    return rows.reset_index(drop=True)


def _class_losses(probabilities):
    clipped = np.clip(probabilities, 1e-7, 1.0 - 1e-7)
    targets = np.asarray([CLASS_TARGETS[label] for label in CLASSES])
    losses = []
    for target in targets:
        losses.append(
            -(
                target[None, :] * np.log(clipped)
                + (1.0 - target[None, :]) * np.log(1.0 - clipped)
            ).mean(axis=1)
        )
    return np.column_stack(losses)


def _best_calibration_setting(targets, losses):
    if not np.any(targets == 1):
        raise ValueError("Calibration groups contain no positive monocyte targets.")
    best = None
    for temperature in TEMPERATURES:
        scores = _monocyte_scores_from_losses(losses, temperature)
        candidates = np.unique(
            np.concatenate(([0.0], scores, [np.nextafter(1.0, 2.0)]))
        )
        for threshold in candidates:
            predicted = scores >= threshold
            _, _, f1, _ = precision_recall_fscore_support(
                targets, predicted, average="binary", zero_division=0
            )
            key = (float(f1), -abs(float(threshold) - 0.5), -temperature)
            if best is None or key > best["key"]:
                best = {
                    "key": key,
                    "temperature": float(temperature),
                    "threshold": float(threshold),
                    "calibration_f1": float(f1),
                }
    if best is None:
        raise RuntimeError("Could not select a calibration setting.")
    return best


def _monocyte_scores_from_losses(losses, temperature):
    scaled = -temperature * (losses - losses.min(axis=1, keepdims=True))
    exponentials = np.exp(scaled)
    return exponentials[:, CLASSES.index("monocyte")] / exponentials.sum(axis=1)


def _metric_summary(targets, predictions):
    precision, recall, f1, _ = precision_recall_fscore_support(
        targets, predictions, average="binary", zero_division=0
    )
    has_positive_targets = bool(np.any(targets == 1))
    return {
        "accuracy": float(accuracy_score(targets, predictions)),
        "precision": float(precision) if has_positive_targets else None,
        "recall": float(recall) if has_positive_targets else None,
        "f1": float(f1) if has_positive_targets else None,
        "positive_support": int(np.sum(targets == 1)),
        "note": (
            None
            if has_positive_targets
            else "Undefined: evaluation groups have no positive weak targets."
        ),
    }


def _group_validation_split(rows, seed):
    labels = rows["label"].to_numpy()
    groups = rows["group_id"].astype(str).to_numpy()
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    _, validation_indices = next(
        splitter.split(np.zeros(len(rows)), labels, groups)
    )
    return rows.iloc[validation_indices].copy().reset_index(drop=True)


def _update_hybrid_outputs(predictions, scores, threshold):
    output = predictions.copy()
    binary_columns = []
    for feature in FEATURES:
        column = f"hybrid_{feature}_binary"
        original_column = f"{feature}_binary"
        if feature in MONOCYTE_DERIVED_FEATURES:
            output[column] = (scores >= threshold).astype(np.int32)
            output[f"hybrid_{feature}_source"] = (
                "thresholded_monocyte_class_likelihood"
            )
        elif original_column in output.columns:
            output[column] = output[original_column].astype(np.int32)
            output[f"hybrid_{feature}_source"] = "original_cnn_binary"
        else:
            raise ValueError(
                f"Prediction file is missing binary column {original_column}."
            )
        binary_columns.append(column)
    output["hybrid_monocyte_class_likelihood"] = scores
    output["hybrid_feature_vector"] = [
        json.dumps(vector, separators=(",", ":"))
        for vector in output[binary_columns].to_numpy(dtype=np.int32).tolist()
    ]
    return output


def create_optimized_hybrid(
    manifest_path=DEFAULT_MANIFEST,
    processed_manifest_path=DEFAULT_PROCESSED_MANIFEST,
    predictions_path=DEFAULT_PREDICTIONS,
    output_predictions_path=DEFAULT_OUTPUT_PREDICTIONS,
    output_report_path=DEFAULT_OUTPUT_REPORT,
    output_summary_path=DEFAULT_OUTPUT_SUMMARY,
    seed=42,
):
    predictions = pd.read_csv(predictions_path)
    rows = _read_matched_rows(manifest_path, processed_manifest_path, predictions)
    validation = _group_validation_split(rows, seed)

    probability_columns = [f"{feature}_probability" for feature in FEATURES]
    probabilities = predictions[probability_columns].to_numpy(dtype=float)
    if not np.isfinite(probabilities).all():
        raise ValueError("Feature probabilities contain non-finite values.")
    if ((probabilities < 0.0) | (probabilities > 1.0)).any():
        raise ValueError("Feature probabilities must be in [0, 1].")
    all_losses = _class_losses(probabilities)
    loss_by_filename = pd.Series(
        list(all_losses), index=predictions["image_filename"]
    )
    validation_losses = np.vstack(
        validation["image_filename"].map(loss_by_filename).to_numpy()
    )
    validation_targets = (
        validation["label"].to_numpy() == "monocyte"
    ).astype(np.int32)
    validation_groups = validation["group_id"].astype(str).to_numpy()
    cv_seed = seed + 1984
    cross_validator = StratifiedGroupKFold(
        n_splits=5, shuffle=True, random_state=cv_seed
    )
    oof_predictions = np.zeros(len(validation), dtype=np.int32)
    fold_settings = []
    for calibration_indices, evaluation_indices in cross_validator.split(
        validation_losses, validation_targets, validation_groups
    ):
        setting = _best_calibration_setting(
            validation_targets[calibration_indices],
            validation_losses[calibration_indices],
        )
        evaluation_scores = _monocyte_scores_from_losses(
            validation_losses[evaluation_indices], setting["temperature"]
        )
        oof_predictions[evaluation_indices] = (
            evaluation_scores >= setting["threshold"]
        ).astype(np.int32)
        fold_settings.append(
            {
                "temperature": setting["temperature"],
                "threshold": setting["threshold"],
                "calibration_f1": setting["calibration_f1"],
                "calibration_samples": int(len(calibration_indices)),
                "evaluation_samples": int(len(evaluation_indices)),
                "calibration_groups": int(
                    len(set(validation_groups[calibration_indices]))
                ),
                "evaluation_groups": int(
                    len(set(validation_groups[evaluation_indices]))
                ),
            }
        )

    final_setting = _best_calibration_setting(
        validation_targets, validation_losses
    )
    all_scores = _monocyte_scores_from_losses(
        all_losses, final_setting["temperature"]
    )
    output = _update_hybrid_outputs(
        predictions, all_scores, final_setting["threshold"]
    )
    output_path = _resolve_path(output_predictions_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    report_path = _resolve_path(output_report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    feature_metrics = {}
    validation_by_filename = validation.set_index("image_filename")
    original_by_filename = predictions.set_index("image_filename")
    for index, feature in enumerate(FEATURES):
        targets = np.asarray(
            [CLASS_TARGETS[label][index] for label in validation_by_filename.label],
            dtype=np.int32,
        )
        filenames = validation_by_filename.index
        baseline = original_by_filename.loc[
            filenames, f"{feature}_binary"
        ].to_numpy(dtype=np.int32)
        if feature in MONOCYTE_DERIVED_FEATURES:
            hybrid = oof_predictions.copy()
        else:
            hybrid = baseline.copy()
        feature_metrics[feature] = {
            "baseline_cnn": _metric_summary(targets, baseline),
            "hybrid_cross_validated": _metric_summary(targets, hybrid),
            "source": (
                "cross-validated_thresholded_monocyte_class_likelihood"
                if feature in MONOCYTE_DERIVED_FEATURES
                else "original_cnn_binary"
            ),
        }

    baseline_feature_f1 = [
        metrics["baseline_cnn"]["f1"]
        for metrics in feature_metrics.values()
        if metrics["baseline_cnn"]["f1"] is not None
    ]
    hybrid_feature_f1 = [
        metrics["hybrid_cross_validated"]["f1"]
        for metrics in feature_metrics.values()
        if metrics["hybrid_cross_validated"]["f1"] is not None
    ]
    summary_rows = []
    for feature, metrics in feature_metrics.items():
        baseline = metrics["baseline_cnn"]
        hybrid = metrics["hybrid_cross_validated"]
        summary_rows.append(
            {
                "feature": feature,
                "prediction_source": metrics["source"],
                "validation_positive_support": hybrid["positive_support"],
                "baseline_precision": baseline["precision"],
                "baseline_recall": baseline["recall"],
                "baseline_f1": baseline["f1"],
                "hybrid_precision": hybrid["precision"],
                "hybrid_recall": hybrid["recall"],
                "hybrid_f1": hybrid["f1"],
                "evaluation_method": "group-stratified 5-fold out-of-fold",
                "target_source": "class-derived weak supervision",
            }
        )
    summary_path = _resolve_path(output_summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_records(summary_rows).to_csv(summary_path, index=False)

    baseline_class_predictions = (
        validation["image_filename"]
        .map(original_by_filename["predicted_wbc_class"])
        .eq("monocyte")
        .to_numpy(dtype=np.int32)
    )
    class_likelihood_metrics = {
        "baseline_cnn_predicted_class": _metric_summary(
            validation_targets, baseline_class_predictions
        ),
        "cross_validated_class_likelihood": _metric_summary(
            validation_targets, oof_predictions
        ),
    }
    report = {
        "dataset": "raabin",
        "target_source": (
            "Class-derived weak supervision; this report does not verify "
            "independently observed cell morphology."
        ),
        "method": (
            "Within each fold of a group-stratified 5-fold split of the original "
            "validation fold, select the temperature and monocyte cutoff using "
            "the other four folds. Aggregate only out-of-fold predictions for "
            "the validation F1 report. Fit a final setting on the full "
            "validation fold for the exported prediction file."
        ),
        "cross_validation": {
            "method": "StratifiedGroupKFold",
            "folds": 5,
            "seed": cv_seed,
            "samples": int(len(validation)),
            "groups": int(validation["group_id"].nunique()),
            "fold_settings": fold_settings,
            "groups_overlap_between_calibration_and_evaluation": False,
        },
        "final_export_calibration": {
            "temperature": final_setting["temperature"],
            "threshold": final_setting["threshold"],
            "calibration_f1": final_setting["calibration_f1"],
            "samples": int(len(validation)),
            "groups": int(validation["group_id"].nunique()),
            "temperature_candidates": list(TEMPERATURES),
        },
        "monocyte_class_likelihood_cross_validated": class_likelihood_metrics,
        "feature_metrics": feature_metrics,
        "feature_f1_summary_csv": str(summary_path.resolve()),
        "macro_f1_over_features_with_positive_weak_targets": {
            "baseline_cnn": float(np.mean(baseline_feature_f1)),
            "optimized_hybrid": float(np.mean(hybrid_feature_f1)),
            "feature_count": len(hybrid_feature_f1),
            "evaluation_method": "pooled out-of-fold predictions",
        },
        "predictions_written": int(len(output)),
        "hybrid_predictions_path": str(output_path.resolve()),
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved optimized hybrid predictions: {output_path}")
    print(f"Saved validation report: {report_path}")
    print(
        f"Group-stratified cross-validation: {len(validation)} images / "
        f"{validation['group_id'].nunique()} groups across 5 folds"
    )
    print(
        f"Final export setting: temperature={final_setting['temperature']:g}, "
        f"threshold={final_setting['threshold']:.4f}; "
        f"cross-validated monocyte F1 "
        f"{class_likelihood_metrics['baseline_cnn_predicted_class']['f1']:.3f} -> "
        f"{class_likelihood_metrics['cross_validated_class_likelihood']['f1']:.3f}"
    )
    print(
        "Macro F1 over features with positive weak targets: "
        f"{np.mean(baseline_feature_f1):.3f} -> "
        f"{np.mean(hybrid_feature_f1):.3f}"
    )
    for feature in MONOCYTE_DERIVED_FEATURES:
        metrics = feature_metrics[feature]
        print(
            f"{feature}: cross-validated F1 "
            f"{metrics['baseline_cnn']['f1']:.3f} -> "
            f"{metrics['hybrid_cross_validated']['f1']:.3f}"
        )
    print(f"Saved 10-feature F1 summary: {summary_path}")
    return report


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Create a group-calibrated monocyte-likelihood hybrid feature file."
        )
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--processed-manifest",
        type=Path,
        default=DEFAULT_PROCESSED_MANIFEST,
    )
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument(
        "--output-predictions", type=Path, default=DEFAULT_OUTPUT_PREDICTIONS
    )
    parser.add_argument("--output-report", type=Path, default=DEFAULT_OUTPUT_REPORT)
    parser.add_argument(
        "--output-summary", type=Path, default=DEFAULT_OUTPUT_SUMMARY
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    create_optimized_hybrid(
        manifest_path=args.manifest,
        processed_manifest_path=args.processed_manifest,
        predictions_path=args.predictions,
        output_predictions_path=args.output_predictions,
        output_report_path=args.output_report,
        output_summary_path=args.output_summary,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
