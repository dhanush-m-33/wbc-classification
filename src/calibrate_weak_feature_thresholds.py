"""Calibrate feature cutoffs against class-derived weak targets.

Run from the project root:
    python src/calibrate_weak_feature_thresholds.py

The original group-separated validation split is divided into group-disjoint
calibration and evaluation subsets. Feature cutoffs are selected on calibration
only; reported calibrated metrics use the untouched evaluation subset.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
)
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
    PROJECT_ROOT / "results" / "raabin_train_feature_predictions_calibrated.csv"
)
DEFAULT_OUTPUT_REPORT = (
    PROJECT_ROOT / "reports" / "raabin_feature_threshold_calibration.json"
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


def _resolve_path(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_accepted_predictions(manifest_path, processed_manifest_path, predictions_path):
    manifest = pd.read_csv(manifest_path)
    processed = pd.read_csv(processed_manifest_path)
    predictions = pd.read_csv(predictions_path)
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

    probability_columns = [f"{feature}_probability" for feature in FEATURES]
    missing_probabilities = sorted(set(probability_columns).difference(predictions.columns))
    if missing_probabilities:
        raise ValueError(
            f"Prediction file is missing feature probability columns: "
            f"{missing_probabilities}"
        )
    accepted = rows.merge(
        predictions[
            ["image_filename", "actual_wbc_class", *probability_columns]
        ],
        on="image_filename",
        how="inner",
        validate="one_to_one",
    )
    class_mismatch = accepted["label"] != accepted["actual_wbc_class"]
    if class_mismatch.any():
        raise ValueError(
            f"{int(class_mismatch.sum())} prediction classes do not match "
            "the training manifest."
        )
    if len(accepted) != len(rows):
        raise ValueError(
            f"Only {len(accepted)} of {len(rows)} accepted training images have "
            "predictions."
        )
    accepted = accepted.reset_index(drop=True)
    accepted["targets"] = accepted["label"].map(CLASS_TARGETS)
    return accepted, predictions


def _split_validation(accepted, seed):
    labels = accepted["label"].to_numpy()
    groups = accepted["group_id"].astype(str).to_numpy()
    split = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    _, validation_indices = next(
        split.split(np.zeros(len(accepted)), labels, groups)
    )
    validation = accepted.iloc[validation_indices].copy().reset_index(drop=True)
    inner_split = StratifiedGroupKFold(
        n_splits=2, shuffle=True, random_state=seed + 1
    )
    _, holdout_indices = next(
        inner_split.split(
            np.zeros(len(validation)),
            validation["label"].to_numpy(),
            validation["group_id"].astype(str).to_numpy(),
        )
    )
    holdout_groups = set(
        validation.iloc[holdout_indices]["group_id"].astype(str)
    )
    validation["calibration_partition"] = np.where(
        validation["group_id"].astype(str).isin(holdout_groups),
        "evaluation",
        "calibration",
    )
    calibration_groups = set(
        validation.loc[
            validation["calibration_partition"] == "calibration", "group_id"
        ].astype(str)
    )
    if calibration_groups.intersection(holdout_groups):
        raise RuntimeError("A source group appears in calibration and evaluation.")
    return validation


def _targets_for(rows, feature_index):
    return np.asarray(
        [CLASS_TARGETS[label][feature_index] for label in rows["label"]],
        dtype=np.int32,
    )


def _best_f1_threshold(targets, probabilities):
    if not np.any(targets == 1):
        return None
    candidates = np.arange(0.05, 0.951, 0.005)
    scores = []
    for threshold in candidates:
        predictions = probabilities >= threshold
        _, _, f1, _ = precision_recall_fscore_support(
            targets, predictions, average="binary", zero_division=0
        )
        scores.append(float(f1))
    best_score = max(scores)
    best_candidates = [
        float(threshold)
        for threshold, score in zip(candidates, scores)
        if np.isclose(score, best_score)
    ]
    return min(best_candidates, key=lambda threshold: abs(threshold - 0.5))


def _metric_summary(targets, probabilities, threshold):
    predictions = probabilities >= threshold
    precision, recall, f1, _ = precision_recall_fscore_support(
        targets, predictions, average="binary", zero_division=0
    )
    has_positive_targets = bool(np.any(targets == 1))
    return {
        "accuracy": float(accuracy_score(targets, predictions)),
        "balanced_accuracy": (
            float(balanced_accuracy_score(targets, predictions))
            if np.unique(targets).size == 2
            else None
        ),
        "precision": float(precision) if has_positive_targets else None,
        "recall": float(recall) if has_positive_targets else None,
        "f1": float(f1) if has_positive_targets else None,
        "positive_support": int(np.sum(targets == 1)),
        "note": (
            None
            if has_positive_targets
            else "Undefined: evaluation subset has no positive weak targets."
        ),
    }


def _confidence_label(probability, threshold):
    if probability < threshold - 0.1:
        return "high_confidence_negative"
    if probability < threshold:
        return "low_confidence_negative"
    if probability < threshold + 0.1:
        return "low_confidence_positive"
    return "high_confidence_positive"


def _write_calibrated_predictions(predictions, thresholds, output_path):
    calibrated = predictions.copy()
    binary_vectors = []
    confidence_vectors = []
    for feature in FEATURES:
        probability_column = f"{feature}_probability"
        probabilities = calibrated[probability_column].to_numpy(dtype=float)
        threshold = thresholds[feature]
        binary = (probabilities >= threshold).astype(np.int32)
        confidence = [
            _confidence_label(float(probability), threshold)
            for probability in probabilities
        ]
        calibrated[f"{feature}_binary"] = binary
        calibrated[f"{feature}_confidence"] = confidence
        binary_vectors.append(binary)
        confidence_vectors.append(confidence)

    calibrated["rounded_binary_feature_vector"] = [
        json.dumps(vector, separators=(",", ":"))
        for vector in np.asarray(binary_vectors, dtype=np.int32).T.tolist()
    ]
    calibrated["confidence_aware_feature_vector"] = [
        json.dumps(vector, separators=(",", ":"))
        for vector in np.asarray(confidence_vectors, dtype=object).T.tolist()
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    calibrated.to_csv(output_path, index=False)
    return calibrated


def calibrate(
    manifest_path=DEFAULT_MANIFEST,
    processed_manifest_path=DEFAULT_PROCESSED_MANIFEST,
    predictions_path=DEFAULT_PREDICTIONS,
    output_predictions_path=DEFAULT_OUTPUT_PREDICTIONS,
    output_report_path=DEFAULT_OUTPUT_REPORT,
    seed=42,
):
    accepted, predictions = _read_accepted_predictions(
        manifest_path, processed_manifest_path, predictions_path
    )
    validation = _split_validation(accepted, seed)
    calibration = validation.loc[
        validation["calibration_partition"] == "calibration"
    ]
    evaluation = validation.loc[
        validation["calibration_partition"] == "evaluation"
    ]

    thresholds = {}
    feature_metrics = {}
    for index, feature in enumerate(FEATURES):
        probability_column = f"{feature}_probability"
        calibration_targets = _targets_for(calibration, index)
        evaluation_targets = _targets_for(evaluation, index)
        calibration_probabilities = calibration[probability_column].to_numpy(
            dtype=float
        )
        evaluation_probabilities = evaluation[probability_column].to_numpy(
            dtype=float
        )
        threshold = _best_f1_threshold(
            calibration_targets, calibration_probabilities
        )
        if threshold is None:
            threshold = 0.5
            note = (
                "Not calibratable: no positive class-derived targets exist in "
                "calibration data; retained the 0.5 cutoff."
            )
        else:
            note = None
        thresholds[feature] = threshold
        feature_metrics[feature] = {
            "threshold": threshold,
            "note": note,
            "calibration": _metric_summary(
                calibration_targets, calibration_probabilities, threshold
            ),
            "evaluation_at_0_5": _metric_summary(
                evaluation_targets, evaluation_probabilities, 0.5
            ),
            "evaluation_at_calibrated_threshold": _metric_summary(
                evaluation_targets, evaluation_probabilities, threshold
            ),
        }

    baseline_f1 = [
        metrics["evaluation_at_0_5"]["f1"]
        for metrics in feature_metrics.values()
        if metrics["evaluation_at_0_5"]["f1"] is not None
    ]
    calibrated_f1 = [
        metrics["evaluation_at_calibrated_threshold"]["f1"]
        for metrics in feature_metrics.values()
        if metrics["evaluation_at_calibrated_threshold"]["f1"] is not None
    ]
    calibrated = _write_calibrated_predictions(
        predictions, thresholds, _resolve_path(output_predictions_path)
    )
    report = {
        "dataset": "raabin",
        "target_source": (
            "Class-derived weak supervision; these metrics do not validate "
            "independently observed morphology."
        ),
        "selection_metric": "Per-feature F1, selected on calibration subset.",
        "validation_split": {
            "method": "StratifiedGroupKFold",
            "folds": 5,
            "seed": seed,
            "samples": int(len(validation)),
            "calibration_samples": int(len(calibration)),
            "evaluation_samples": int(len(evaluation)),
            "calibration_evaluation_groups_overlap": False,
        },
        "thresholds": thresholds,
        "feature_metrics": feature_metrics,
        "macro_f1_over_features_with_positive_targets": {
            "evaluation_at_0_5": float(np.mean(baseline_f1)),
            "evaluation_at_calibrated_threshold": float(np.mean(calibrated_f1)),
            "feature_count": len(calibrated_f1),
        },
        "predictions_written": int(len(calibrated)),
        "calibrated_predictions_path": str(
            _resolve_path(output_predictions_path).resolve()
        ),
    }
    report_path = _resolve_path(output_report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved calibrated predictions: {_resolve_path(output_predictions_path)}")
    print(f"Saved calibration report: {report_path}")
    print(
        f"Group-disjoint calibration/evaluation images: "
        f"{len(calibration)}/{len(evaluation)}"
    )
    print(
        "Evaluation macro F1 across features with positive weak targets: "
        f"{np.mean(baseline_f1):.3f} -> {np.mean(calibrated_f1):.3f}"
    )
    for feature, metrics in feature_metrics.items():
        baseline = metrics["evaluation_at_0_5"]["f1"]
        calibrated_f1 = metrics["evaluation_at_calibrated_threshold"]["f1"]
        baseline_text = "n/a" if baseline is None else f"{baseline:.3f}"
        calibrated_text = (
            "n/a" if calibrated_f1 is None else f"{calibrated_f1:.3f}"
        )
        print(
            f"{feature}: threshold={metrics['threshold']:.3f}, "
            f"evaluation F1 {baseline_text} -> {calibrated_text}"
        )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate per-feature prediction cutoffs using group-separated "
            "class-derived weak targets."
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
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    calibrate(
        manifest_path=args.manifest,
        processed_manifest_path=args.processed_manifest,
        predictions_path=args.predictions,
        output_predictions_path=args.output_predictions,
        output_report_path=args.output_report,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
