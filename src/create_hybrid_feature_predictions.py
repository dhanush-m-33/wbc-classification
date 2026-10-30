"""Create a class-derived hybrid feature prediction file.

Run from the project root:
    python src/create_hybrid_feature_predictions.py

The three monocyte-associated feature binaries use the CNN's predicted WBC
class. Thin-rim and all other feature binaries remain the original CNN outputs.
Metrics compare against the existing class-derived weak targets, not human
morphology annotations.
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
    PROJECT_ROOT / "results" / "raabin_train_feature_predictions_hybrid.csv"
)
DEFAULT_OUTPUT_REPORT = (
    PROJECT_ROOT / "reports" / "raabin_hybrid_feature_metrics.json"
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
MONOCYTE_DERIVED_FEATURES = (
    "abundant_cytoplasm",
    "kidney_bean_shaped_nucleus",
    "eccentric_and_large_nucleus",
)
CLASSES = ("eosinophil", "lymphocyte", "monocyte", "neutrophil")


def _resolve_path(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_validation_rows(manifest_path, processed_manifest_path, predictions):
    manifest = pd.read_csv(manifest_path)
    processed = pd.read_csv(processed_manifest_path)
    required_manifest = {"path", "label", "dataset", "split", "group_id"}
    required_processed = {"raw_path", "path", "label"}
    required_predictions = {
        "image_filename",
        "actual_wbc_class",
        "predicted_wbc_class",
    }
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
    predictions["predicted_wbc_class"] = (
        predictions["predicted_wbc_class"].astype(str).str.strip().str.lower()
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

    rows = rows.merge(
        predictions[
            ["image_filename", "actual_wbc_class", "predicted_wbc_class"]
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
        raise ValueError("No accepted training predictions were matched.")
    return rows.reset_index(drop=True)


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
            else "Undefined: no positive class-derived targets exist."
        ),
    }


def create_hybrid_predictions(
    manifest_path=DEFAULT_MANIFEST,
    processed_manifest_path=DEFAULT_PROCESSED_MANIFEST,
    predictions_path=DEFAULT_PREDICTIONS,
    output_predictions_path=DEFAULT_OUTPUT_PREDICTIONS,
    output_report_path=DEFAULT_OUTPUT_REPORT,
    seed=42,
):
    predictions = pd.read_csv(predictions_path)
    required_columns = {
        "image_filename",
        "predicted_wbc_class",
        *[f"{feature}_binary" for feature in FEATURES],
    }
    missing_columns = sorted(required_columns.difference(predictions.columns))
    if missing_columns:
        raise ValueError(
            f"Prediction file is missing required columns: {missing_columns}"
        )

    rows = _read_validation_rows(manifest_path, processed_manifest_path, predictions)
    labels = rows["label"].to_numpy()
    groups = rows["group_id"].astype(str).to_numpy()
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    _, validation_indices = next(
        splitter.split(np.zeros(len(rows)), labels, groups)
    )
    validation = rows.iloc[validation_indices].copy()
    if len(validation) != len(validation_indices):
        raise RuntimeError("Validation row alignment failed.")

    original_by_filename = predictions.set_index("image_filename")
    hybrid_by_filename = predictions.copy().set_index("image_filename")
    sources = {}
    for feature in FEATURES:
        original_column = f"{feature}_binary"
        hybrid_column = f"hybrid_{feature}_binary"
        source_column = f"hybrid_{feature}_source"
        if feature in MONOCYTE_DERIVED_FEATURES:
            binary = (
                original_by_filename["predicted_wbc_class"] == "monocyte"
            ).astype(np.int32)
            source = "predicted_wbc_class==monocyte"
        else:
            binary = original_by_filename[original_column].astype(np.int32)
            source = "cnn_probability>=0.5"
        hybrid_by_filename[hybrid_column] = binary
        hybrid_by_filename[source_column] = source
        sources[feature] = source

    hybrid_columns = [f"hybrid_{feature}_binary" for feature in FEATURES]
    hybrid_by_filename["hybrid_feature_vector"] = [
        json.dumps(vector, separators=(",", ":"))
        for vector in hybrid_by_filename[hybrid_columns]
        .to_numpy(dtype=np.int32)
        .tolist()
    ]
    output = predictions.merge(
        hybrid_by_filename[
            [
                *hybrid_columns,
                *[f"hybrid_{feature}_source" for feature in FEATURES],
                "hybrid_feature_vector",
            ]
        ],
        left_on="image_filename",
        right_index=True,
        how="left",
        validate="one_to_one",
    )
    if output[hybrid_columns].isna().any().any():
        raise ValueError("Hybrid outputs were not produced for every prediction row.")

    metrics = {}
    validation_by_filename = validation.set_index("image_filename")
    for index, feature in enumerate(FEATURES):
        targets = np.asarray(
            [
                CLASS_TARGETS[label][index]
                for label in validation_by_filename["label"]
            ],
            dtype=np.int32,
        )
        filenames = validation_by_filename.index
        original = original_by_filename.loc[
            filenames, f"{feature}_binary"
        ].to_numpy(dtype=np.int32)
        hybrid = hybrid_by_filename.loc[
            filenames, f"hybrid_{feature}_binary"
        ].to_numpy(dtype=np.int32)
        metrics[feature] = {
            "source": sources[feature],
            "baseline_cnn": _metric_summary(targets, original),
            "hybrid": _metric_summary(targets, hybrid),
        }

    for feature in MONOCYTE_DERIVED_FEATURES:
        if metrics[feature]["hybrid"]["f1"] is None:
            raise RuntimeError(f"Hybrid F1 is undefined for {feature}.")
        if (
            metrics[feature]["hybrid"]["f1"]
            <= metrics[feature]["baseline_cnn"]["f1"]
        ):
            raise RuntimeError(
                f"Hybrid predictions did not improve validation F1 for {feature}."
            )

    output_path = _resolve_path(output_predictions_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    report_path = _resolve_path(output_report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "dataset": "raabin",
        "target_source": (
            "Class-derived weak supervision; this report does not validate "
            "independently observed morphology."
        ),
        "method": (
            "Use the CNN's predicted WBC class as a proxy for the three "
            "monocyte-associated feature binaries; leave other CNN binaries "
            "unchanged."
        ),
        "validation_split": {
            "method": "StratifiedGroupKFold",
            "folds": 5,
            "seed": seed,
            "samples": int(len(validation)),
            "groups": int(validation["group_id"].nunique()),
        },
        "feature_metrics": metrics,
        "predictions_written": int(len(output)),
        "hybrid_predictions_path": str(output_path.resolve()),
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved hybrid predictions: {output_path}")
    print(f"Saved validation report: {report_path}")
    print(f"Validation images: {len(validation)}")
    for feature in MONOCYTE_DERIVED_FEATURES:
        before = metrics[feature]["baseline_cnn"]["f1"]
        after = metrics[feature]["hybrid"]["f1"]
        print(f"{feature}: validation F1 {before:.3f} -> {after:.3f}")


def main():
    parser = argparse.ArgumentParser(
        description="Create and evaluate hybrid RAABIN feature predictions."
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
    create_hybrid_predictions(
        manifest_path=args.manifest,
        processed_manifest_path=args.processed_manifest,
        predictions_path=args.predictions,
        output_predictions_path=args.output_predictions,
        output_report_path=args.output_report,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
