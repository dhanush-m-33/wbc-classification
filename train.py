import argparse
import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import StratifiedGroupKFold


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "manifests" / "raabin_train.csv"
DEFAULT_PROCESSED_MANIFEST = (
    PROJECT_ROOT / "data" / "manifests" / "processed_raabin_train.csv"
)
PROCESSED_TRAIN_DIR = PROJECT_ROOT / "data" / "processed" / "raabin" / "train"
MODEL_PATH = PROJECT_ROOT / "models" / "raabin_morphology_cnn.keras"
REPORTS_DIR = PROJECT_ROOT / "reports"
RESULTS_DIR = PROJECT_ROOT / "results"
FEATURE_MAPPING_PATH = (
    PROJECT_ROOT / "features" / "raabin_morphology_feature_mapping.json"
)
SEED = 42
IMAGE_SIZE = 64
BATCH_SIZE = 128
MAX_EPOCHS = 25
EARLY_STOPPING_PATIENCE = 5
VALIDATION_FOLDS = 5

FEATURES = (
    ("granules", "Granules"),
    ("dense_pink_granules", "Dense pink granules"),
    ("dense_blue_granules", "Dense blue granules"),
    ("thin_rim_of_cytoplasm", "Thin rim of cytoplasm"),
    ("abundant_cytoplasm", "Abundant cytoplasm"),
    ("multi_lobed_nucleus_3_5", "Multi-lobed (3-5) nucleus"),
    ("bilobed_nucleus", "Bilobed nucleus"),
    ("kidney_bean_shaped_nucleus", "Kidney bean-shaped nucleus"),
    ("eccentric_and_large_nucleus", "Eccentric and large nucleus"),
    ("small_size_dense_cytoplasm", "Small size / Same as RBC, dense cytoplasm"),
)
FEATURE_NAMES = [name for name, _ in FEATURES]
CLASSES = ("eosinophil", "lymphocyte", "monocyte", "neutrophil")
CLASS_TARGETS = {
    "eosinophil": [1, 1, 0, 0, 0, 0, 1, 0, 0, 0],
    "lymphocyte": [0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
    "monocyte": [0, 0, 0, 0, 1, 0, 0, 1, 1, 0],
    "neutrophil": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
}


def _resolve_path(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_manifests(manifest_path, processed_manifest_path):
    manifest = pd.read_csv(manifest_path)
    processed = pd.read_csv(processed_manifest_path)
    required_manifest = {"path", "label", "dataset", "split", "group_id"}
    required_processed = {"raw_path", "path", "qc_flag", "label"}
    missing_manifest = sorted(required_manifest.difference(manifest.columns))
    missing_processed = sorted(required_processed.difference(processed.columns))
    if missing_manifest:
        raise ValueError(f"Training manifest is missing columns: {missing_manifest}")
    if missing_processed:
        raise ValueError(
            f"Processed manifest is missing columns: {missing_processed}"
        )

    manifest["label"] = manifest["label"].astype(str).str.strip().str.lower()
    if not manifest["dataset"].astype(str).str.lower().eq("raabin").all():
        raise ValueError("The training manifest contains non-RAABIN rows.")
    if not manifest["split"].astype(str).str.lower().eq("train").all():
        raise ValueError("The training manifest must contain only the train split.")
    unknown_classes = sorted(set(manifest["label"]).difference(CLASSES))
    if unknown_classes:
        raise ValueError(f"Unsupported WBC classes in manifest: {unknown_classes}")

    processed["label"] = processed["label"].astype(str).str.strip().str.lower()
    rows = manifest.merge(
        processed[["raw_path", "path", "qc_flag", "label"]].rename(
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
            "the primary training manifest."
        )

    missing_raw_paths = []
    missing_processed_paths = []
    processed_image_paths = []
    qc_status = []
    for row in rows.itertuples(index=False):
        raw_path = _resolve_path(row.path)
        if not raw_path.is_file():
            missing_raw_paths.append(str(raw_path))

        if isinstance(row.processed_path, str) and row.processed_path:
            image_path = _resolve_path(row.processed_path)
            status = str(row.qc_flag)
        else:
            image_path = (
                PROCESSED_TRAIN_DIR
                / row.label
                / f"{Path(row.path).stem}.png"
            )
            status = "not_in_accepted_processed_manifest"
        if image_path.is_file():
            processed_image_paths.append(str(image_path.resolve()))
        else:
            missing_processed_paths.append(str(image_path))
            processed_image_paths.append(None)
        qc_status.append(status)

    rows["image_path"] = processed_image_paths
    rows["preprocessing_qc"] = qc_status
    training_rows = rows.loc[matched & rows["image_path"].notna()].copy()
    if training_rows.empty:
        raise ValueError("No matching QC-accepted processed training images were found.")

    return rows, training_rows, missing_raw_paths, missing_processed_paths


def _decode_image(path):
    image = tf.io.decode_png(tf.io.read_file(path), channels=3)
    image = tf.ensure_shape(image, (IMAGE_SIZE, IMAGE_SIZE, 3))
    return tf.cast(image, tf.float32) / 255.0


def _make_model():
    inputs = tf.keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3), name="image")
    x = inputs
    for filters in (24, 32, 48):
        x = tf.keras.layers.SeparableConv2D(
            filters,
            kernel_size=3,
            strides=2,
            padding="same",
            use_bias=False,
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("swish")(x)
    x = tf.keras.layers.SeparableConv2D(
        64, kernel_size=3, padding="same", use_bias=False
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("swish")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    outputs = tf.keras.layers.Dense(
        10, activation="sigmoid", name="morphology_features"
    )(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="wbc_morphology_cnn")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="binary_accuracy", threshold=0.5),
            tf.keras.metrics.Precision(name="precision", thresholds=0.5),
            tf.keras.metrics.Recall(name="recall", thresholds=0.5),
        ],
    )
    return model


def _make_dataset(paths, targets, sample_weights=None, shuffle=False, seed=SEED):
    if targets is None:
        dataset = tf.data.Dataset.from_tensor_slices(paths)
        dataset = dataset.map(
            _decode_image, num_parallel_calls=tf.data.AUTOTUNE
        )
    elif sample_weights is None:
        dataset = tf.data.Dataset.from_tensor_slices((paths, targets))

        def load_validation_image(path, target):
            return _decode_image(path), target

        dataset = dataset.map(
            load_validation_image, num_parallel_calls=tf.data.AUTOTUNE
        )
    else:
        dataset = tf.data.Dataset.from_tensor_slices(
            (paths, targets, sample_weights)
        )

        def load_labeled_image(path, target, weight):
            return _decode_image(path), target, weight

        dataset = dataset.map(
            load_labeled_image, num_parallel_calls=tf.data.AUTOTUNE
        )
    if shuffle:
        dataset = dataset.shuffle(
            min(len(paths), 4096), seed=seed, reshuffle_each_iteration=True
        )
    return dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


def _feature_output(probabilities):
    binary = (probabilities >= 0.5).astype(np.int32)
    confidence = []
    for probability in probabilities:
        if probability < 0.4:
            confidence.append("high_confidence_negative")
        elif probability < 0.5:
            confidence.append("low_confidence_negative")
        elif probability < 0.6:
            confidence.append("low_confidence_positive")
        else:
            confidence.append("high_confidence_positive")
    return binary, confidence


def predict_new_images(image_paths, model_path=MODEL_PATH, batch_size=BATCH_SIZE):
    """Return continuous feature probabilities and thresholded categories for PNGs."""
    model = tf.keras.models.load_model(model_path)
    paths = [str(Path(path).resolve()) for path in image_paths]
    if not paths:
        return pd.DataFrame(columns=["filename", *FEATURE_NAMES])

    dataset = tf.data.Dataset.from_tensor_slices(paths)
    dataset = dataset.map(_decode_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    probabilities = model.predict(dataset, verbose=0)

    records = []
    for path, vector in zip(paths, probabilities):
        binary, confidence = _feature_output(vector)
        record = {"filename": Path(path).name}
        record.update(
            {f"{name}_probability": float(value) for name, value in zip(FEATURE_NAMES, vector)}
        )
        record.update(
            {f"{name}_binary": int(value) for name, value in zip(FEATURE_NAMES, binary)}
        )
        record.update(
            {f"{name}_confidence": value for name, value in zip(FEATURE_NAMES, confidence)}
        )
        records.append(record)
    return pd.DataFrame.from_records(records)


def _predict_wbc_classes(probabilities):
    targets = np.asarray([CLASS_TARGETS[name] for name in CLASSES], dtype=np.float32)
    clipped = np.clip(probabilities[:, None, :], 1e-7, 1.0 - 1e-7)
    losses = -(
        targets[None, :, :] * np.log(clipped)
        + (1.0 - targets[None, :, :]) * np.log(1.0 - clipped)
    ).mean(axis=2)
    return np.asarray(CLASSES)[losses.argmin(axis=1)]


def _save_prediction_csv(rows, probabilities, output_path):
    records = []
    predicted_classes = _predict_wbc_classes(probabilities)
    for row, vector, predicted_class in zip(
        rows.itertuples(index=False), probabilities, predicted_classes
    ):
        binary, confidence = _feature_output(vector)
        record = {
            "image_filename": Path(row.image_path).name,
            "dataset_name": row.dataset,
            "actual_wbc_class": row.label,
            "predicted_wbc_class": str(predicted_class),
            "preprocessing_qc": row.preprocessing_qc,
            "continuous_feature_vector": json.dumps(
                [float(value) for value in vector], separators=(",", ":")
            ),
            "rounded_binary_feature_vector": json.dumps(
                binary.tolist(), separators=(",", ":")
            ),
            "confidence_aware_feature_vector": json.dumps(
                confidence, separators=(",", ":")
            ),
        }
        record.update(
            {
                f"{name}_probability": float(value)
                for name, value in zip(FEATURE_NAMES, vector)
            }
        )
        record.update(
            {
                f"{name}_binary": int(value)
                for name, value in zip(FEATURE_NAMES, binary)
            }
        )
        record.update(
            {
                f"{name}_confidence": value
                for name, value in zip(FEATURE_NAMES, confidence)
            }
        )
        records.append(record)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_records(records).to_csv(output_path, index=False)
    return predicted_classes


def _save_training_plots(history, reports_dir):
    history_data = history.history
    epochs = range(1, len(history_data["loss"]) + 1)

    loss_path = reports_dir / "raabin_cnn_loss.png"
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(epochs, history_data["loss"], label="Training loss")
    axis.plot(epochs, history_data["val_loss"], label="Validation loss")
    axis.set(title="RAABIN CNN loss", xlabel="Epoch", ylabel="Binary cross-entropy")
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(loss_path, dpi=160)
    plt.close(figure)

    precision = np.asarray(history_data.get("precision", []), dtype=float)
    recall = np.asarray(history_data.get("recall", []), dtype=float)
    val_precision = np.asarray(history_data.get("val_precision", []), dtype=float)
    val_recall = np.asarray(history_data.get("val_recall", []), dtype=float)
    if precision.size and recall.size and val_precision.size and val_recall.size:
        train_f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-8)
        val_f1 = 2 * val_precision * val_recall / np.maximum(
            val_precision + val_recall, 1e-8
        )
        metrics = (
            ("binary_accuracy", "Binary accuracy"),
            ("precision", "Micro precision"),
            ("recall", "Micro recall"),
        )
        figure, axes = plt.subplots(2, 2, figsize=(11, 8))
        for axis, (key, title) in zip(axes.flat[:3], metrics):
            axis.plot(epochs, history_data[key], label="Training")
            axis.plot(epochs, history_data[f"val_{key}"], label="Validation")
            axis.set(title=title, xlabel="Epoch", ylim=(0, 1))
            axis.grid(alpha=0.25)
            axis.legend()
        axes.flat[3].plot(epochs, train_f1, label="Training")
        axes.flat[3].plot(epochs, val_f1, label="Validation")
        axes.flat[3].set(title="Micro F1", xlabel="Epoch", ylim=(0, 1))
        axes.flat[3].grid(alpha=0.25)
        axes.flat[3].legend()
        figure.suptitle("RAABIN CNN validation metrics (weak targets)")
        figure.tight_layout()
        figure.savefig(reports_dir / "raabin_cnn_metrics.png", dpi=160)
        plt.close(figure)


def _feature_metrics(targets, probabilities):
    predictions = (probabilities >= 0.5).astype(np.int32)
    precision, recall, f1, support = precision_recall_fscore_support(
        targets, predictions, average=None, zero_division=0
    )
    results = {}
    for index, name in enumerate(FEATURE_NAMES):
        has_positive_targets = bool(np.any(targets[:, index] == 1))
        results[name] = {
            "precision": float(precision[index]) if has_positive_targets else None,
            "recall": float(recall[index]) if has_positive_targets else None,
            "f1": float(f1[index]) if has_positive_targets else None,
            "support": int(support[index]),
            "note": (
                None
                if has_positive_targets
                else "Undefined: no positive weak targets exist for this feature."
            ),
        }
    return results


def train(
    manifest_path=DEFAULT_MANIFEST,
    processed_manifest_path=DEFAULT_PROCESSED_MANIFEST,
    seed=SEED,
    max_epochs=MAX_EPOCHS,
):
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except RuntimeError:
        pass

    rows, training_rows, missing_raw, missing_processed = _read_manifests(
        manifest_path, processed_manifest_path
    )
    if missing_raw:
        print(f"Missing original manifest image paths: {len(missing_raw)}")
        for path in missing_raw[:10]:
            print(f"  {path}")
    if missing_processed:
        print(f"Missing processed image paths: {len(missing_processed)}")
        for path in missing_processed[:10]:
            print(f"  {path}")

    labels = training_rows["label"].to_numpy()
    groups = training_rows["group_id"].astype(str).to_numpy()
    targets = np.asarray([CLASS_TARGETS[label] for label in labels], dtype=np.float32)
    counts = pd.Series(labels).value_counts()
    missing_classes = sorted(set(CLASSES).difference(counts.index))
    if missing_classes:
        raise ValueError(f"Accepted training data is missing classes: {missing_classes}")
    if counts.min() < VALIDATION_FOLDS:
        raise ValueError("Not enough samples in each class for the grouped validation split.")

    splitter = StratifiedGroupKFold(
        n_splits=VALIDATION_FOLDS, shuffle=True, random_state=seed
    )
    train_indices, validation_indices = next(
        splitter.split(np.zeros(len(labels)), labels, groups)
    )
    train_groups = set(groups[train_indices])
    validation_groups = set(groups[validation_indices])
    if train_groups.intersection(validation_groups):
        raise RuntimeError("A source group appears in both training and validation.")

    paths = training_rows["image_path"].to_numpy(dtype=str)
    class_weights = {
        name: len(train_indices) / (len(CLASSES) * int((labels[train_indices] == name).sum()))
        for name in CLASSES
    }
    sample_weights = np.asarray(
        [class_weights[label] for label in labels[train_indices]], dtype=np.float32
    )
    train_dataset = _make_dataset(
        paths[train_indices],
        targets[train_indices],
        sample_weights,
        shuffle=True,
        seed=seed,
    )
    validation_dataset = _make_dataset(
        paths[validation_indices], targets[validation_indices]
    )

    model = _make_model()
    model.summary()
    device_names = [device.name for device in tf.config.list_physical_devices()]
    print(f"TensorFlow devices: {device_names}")
    print(f"Using {len(train_indices)} training and {len(validation_indices)} validation images.")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        MODEL_PATH,
        monitor="val_loss",
        mode="min",
        save_best_only=True,
        verbose=1,
    )
    callbacks = [
        checkpoint,
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5, verbose=1
        ),
    ]
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=max_epochs,
        callbacks=callbacks,
        verbose=2,
    )
    model.save(MODEL_PATH)

    history_path = REPORTS_DIR / "raabin_cnn_training_history.json"
    history_path.write_text(
        json.dumps(history.history, indent=2), encoding="utf-8"
    )
    pd.DataFrame(history.history).rename_axis("epoch").to_csv(
        REPORTS_DIR / "raabin_cnn_training_history.csv"
    )
    _save_training_plots(history, REPORTS_DIR)

    validation_probabilities = model.predict(validation_dataset, verbose=0)
    validation_targets = targets[validation_indices]
    feature_metrics = _feature_metrics(
        validation_targets, validation_probabilities
    )
    best_epoch = int(np.argmin(history.history["val_loss"]))
    accepted_counts = {name: int(counts.get(name, 0)) for name in CLASSES}

    prediction_paths = rows["image_path"].notna()
    prediction_rows = rows.loc[prediction_paths].copy().reset_index(drop=True)
    prediction_dataset = _make_dataset(
        prediction_rows["image_path"].to_numpy(dtype=str), targets=None
    )
    all_probabilities = model.predict(prediction_dataset, verbose=0)
    prediction_path = RESULTS_DIR / "raabin_train_feature_predictions_latest.csv"
    _save_prediction_csv(prediction_rows, all_probabilities, prediction_path)
    from src.create_optimized_hybrid_feature_predictions import (
        create_optimized_hybrid,
    )

    hybrid_report = create_optimized_hybrid(
        manifest_path=manifest_path,
        processed_manifest_path=processed_manifest_path,
        predictions_path=prediction_path,
        output_predictions_path=(
            RESULTS_DIR / "raabin_train_feature_predictions_hybrid_latest.csv"
        ),
        output_report_path=(
            REPORTS_DIR / "raabin_hybrid_feature_metrics_latest.json"
        ),
        output_summary_path=REPORTS_DIR / "raabin_feature_f1_summary_latest.csv",
        seed=seed,
    )

    FEATURE_MAPPING_PATH.parent.mkdir(parents=True, exist_ok=True)
    feature_mapping = {
        "feature_names": FEATURE_NAMES,
        "feature_descriptions": dict(FEATURES),
        "target_source": "class-based weak supervision; no per-feature annotations present",
        "class_weak_supervision_targets": CLASS_TARGETS,
        "continuous_output": "10 sigmoid probabilities in feature_names order",
        "binary_threshold": 0.5,
        "confidence_thresholds": {
            "probability_below_0_40": "0, high confidence",
            "probability_0_40_to_below_0_50": "0, low confidence",
            "probability_0_50_to_below_0_60": "1, low confidence",
            "probability_at_least_0_60": "1, high confidence",
        },
    }
    FEATURE_MAPPING_PATH.write_text(
        json.dumps(feature_mapping, indent=2), encoding="utf-8"
    )

    metrics = {
        "dataset": "raabin",
        "primary_manifest": str(Path(manifest_path).resolve()),
        "processed_manifest": str(Path(processed_manifest_path).resolve()),
        "processed_images_used_for_training": int(len(training_rows)),
        "prediction_rows_written": int(len(prediction_rows)),
        "missing_original_manifest_paths": missing_raw,
        "missing_processed_image_paths": missing_processed,
        "rows_excluded_from_training_by_qc_or_missing_processed_image": int(
            len(rows) - len(training_rows)
        ),
        "class_counts_accepted_for_training": accepted_counts,
        "class_weights": class_weights,
        "train_samples": int(len(train_indices)),
        "validation_samples": int(len(validation_indices)),
        "train_class_counts": {
            name: int((labels[train_indices] == name).sum()) for name in CLASSES
        },
        "validation_class_counts": {
            name: int((labels[validation_indices] == name).sum())
            for name in CLASSES
        },
        "validation_groups_overlap_training": bool(
            train_groups.intersection(validation_groups)
        ),
        "best_epoch": best_epoch + 1,
        "epochs_completed": len(history.history["loss"]),
        "training_loss_at_best_epoch": float(history.history["loss"][best_epoch]),
        "validation_loss_at_best_epoch": float(
            history.history["val_loss"][best_epoch]
        ),
        "test_evaluated": False,
        "test_loss": None,
        "test_metrics": None,
        "test_evaluation_note": (
            "Not evaluated: the user requested training-only validation until "
            "processed TestA/TestB images are provided."
        ),
        "feature_metric_interpretation": (
            "Validation precision, recall, and F1 measure agreement with "
            "class-derived weak targets, not independently annotated morphology."
        ),
        "validation_feature_metrics_against_weak_targets": feature_metrics,
        "training_history_json": str(
            (REPORTS_DIR / "raabin_cnn_training_history.json").resolve()
        ),
        "loss_plot": str((REPORTS_DIR / "raabin_cnn_loss.png").resolve()),
        "metrics_plot": str((REPORTS_DIR / "raabin_cnn_metrics.png").resolve()),
        "model_path": str(MODEL_PATH.resolve()),
        "prediction_csv": str(prediction_path.resolve()),
        "hybrid_prediction_csv": hybrid_report["hybrid_predictions_path"],
        "hybrid_validation_report": str(
            (REPORTS_DIR / "raabin_hybrid_feature_metrics_latest.json").resolve()
        ),
        "feature_f1_summary_csv": hybrid_report["feature_f1_summary_csv"],
        "feature_mapping": str(FEATURE_MAPPING_PATH.resolve()),
        "feature_target_source": feature_mapping["target_source"],
        "image_size": [IMAGE_SIZE, IMAGE_SIZE],
        "seed": seed,
        "tensorflow_version": tf.__version__,
        "tensorflow_devices": device_names,
    }
    metrics_path = REPORTS_DIR / "raabin_cnn_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Feature targets: {metrics['feature_target_source']}")
    print(f"Validation loss: {metrics['validation_loss_at_best_epoch']:.5f}")
    print(f"Training loss at best epoch: {metrics['training_loss_at_best_epoch']:.5f}")
    print("Independent test metrics: not evaluated (processed test data unavailable).")
    print(f"Saved CNN: {MODEL_PATH}")
    print(f"Saved predictions: {prediction_path} ({len(prediction_rows)} images)")
    print(
        "Saved hybrid predictions and 10-feature F1 summary: "
        f"{hybrid_report['feature_f1_summary_csv']}"
    )
    print(f"Saved metrics: {metrics_path}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Train a lightweight 10-output CNN from existing processed RAABIN "
            "train images using class-based weak supervision."
        )
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--processed-manifest",
        type=Path,
        default=DEFAULT_PROCESSED_MANIFEST,
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    args = parser.parse_args()
    train(
        args.manifest,
        args.processed_manifest,
        seed=args.seed,
        max_epochs=args.epochs,
    )


if __name__ == "__main__":
    main()
