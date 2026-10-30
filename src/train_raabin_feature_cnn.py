"""Train a RAABIN-only CNN against class-derived morphology feature targets."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedGroupKFold


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "manifests" / "raabin_train.csv"
DEFAULT_PROCESSED_MANIFEST = (
    PROJECT_ROOT / "data" / "manifests" / "processed_raabin_train.csv"
)
DEFAULT_MODEL = PROJECT_ROOT / "models" / "raabin_feature_cnn.keras"
DEFAULT_VALIDATION_CSV = (
    PROJECT_ROOT / "results" / "raabin_feature_cnn_validation_with_vectors.csv"
)
DEFAULT_HISTORY_CSV = (
    PROJECT_ROOT / "reports" / "raabin_feature_cnn_history.csv"
)
DEFAULT_METRICS_JSON = (
    PROJECT_ROOT / "reports" / "raabin_feature_cnn_metrics.json"
)
FEATURE_MAPPING = (
    PROJECT_ROOT / "features" / "raabin_morphology_feature_mapping.json"
)
IMAGE_SIZE = 64
BATCH_SIZE = 128
MAX_EPOCHS = 25
SEED = 42
CLASSES = ("eosinophil", "lymphocyte", "monocyte", "neutrophil")


def _resolve_path(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _load_targets():
    with FEATURE_MAPPING.open(encoding="utf-8") as file:
        mapping = json.load(file)
    feature_names = mapping["feature_names"]
    class_targets = mapping["class_weak_supervision_targets"]
    if set(class_targets) != set(CLASSES):
        raise ValueError("Feature mapping must define targets for all RAABIN classes.")
    if any(len(class_targets[name]) != len(feature_names) for name in CLASSES):
        raise ValueError("Feature target vectors do not match feature_names.")
    return feature_names, class_targets


def _load_training_rows(manifest_path, processed_manifest_path):
    manifest = pd.read_csv(manifest_path)
    processed = pd.read_csv(processed_manifest_path)
    manifest_required = {"path", "label", "dataset", "split", "group_id"}
    processed_required = {"raw_path", "path", "label", "qc_flag"}
    for name, frame, required in (
        ("training manifest", manifest, manifest_required),
        ("processed manifest", processed, processed_required),
    ):
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"{name} is missing required columns: {missing}")

    manifest["label"] = manifest["label"].astype(str).str.strip().str.lower()
    processed["label"] = processed["label"].astype(str).str.strip().str.lower()
    if not manifest["dataset"].astype(str).str.lower().eq("raabin").all():
        raise ValueError("Training manifest contains rows outside RAABIN.")
    if not manifest["split"].astype(str).str.lower().eq("train").all():
        raise ValueError("Training manifest must contain only the train split.")
    unknown_classes = sorted(set(manifest["label"]).difference(CLASSES))
    if unknown_classes:
        raise ValueError(f"Unsupported WBC classes: {unknown_classes}")

    rows = manifest.merge(
        processed[["raw_path", "path", "label", "qc_flag"]].rename(
            columns={"path": "processed_path", "label": "processed_label"}
        ),
        left_on="path",
        right_on="raw_path",
        how="left",
        validate="one_to_one",
    )
    matched = rows["processed_path"].notna()
    mismatched_labels = matched & (rows["label"] != rows["processed_label"])
    if mismatched_labels.any():
        raise ValueError(
            f"{int(mismatched_labels.sum())} processed labels differ from "
            "the RAABIN training manifest."
        )

    rows["image_path"] = rows["processed_path"].map(
        lambda value: _resolve_path(value) if isinstance(value, str) else None
    )
    existing = rows["image_path"].map(
        lambda value: isinstance(value, Path) and value.is_file()
    )
    training_rows = rows.loc[matched & existing].copy().reset_index(drop=True)
    if training_rows.empty:
        raise ValueError("No processed RAABIN training images were found.")
    training_rows["image_path"] = training_rows["image_path"].map(str)
    return training_rows, int((~(matched & existing)).sum())


def _decode_image(path, target=None, sample_weight=None):
    image = tf.io.decode_png(tf.io.read_file(path), channels=3)
    image = tf.image.resize(image, (IMAGE_SIZE, IMAGE_SIZE), antialias=True)
    image = tf.cast(image, tf.float32) / 255.0
    if target is None:
        return image
    if sample_weight is None:
        return image, target
    return image, target, sample_weight


def _make_dataset(
    paths, targets=None, sample_weights=None, shuffle=False, seed=SEED
):
    if targets is None:
        dataset = tf.data.Dataset.from_tensor_slices(paths)
        dataset = dataset.map(
            lambda path: _decode_image(path),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    elif sample_weights is None:
        dataset = tf.data.Dataset.from_tensor_slices((paths, targets))
        dataset = dataset.map(
            lambda path, target: _decode_image(path, target),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    else:
        dataset = tf.data.Dataset.from_tensor_slices(
            (paths, targets, sample_weights)
        )
        dataset = dataset.map(
            lambda path, target, weight: _decode_image(path, target, weight),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    if shuffle:
        dataset = dataset.shuffle(
            min(len(paths), 4096), seed=seed, reshuffle_each_iteration=True
        )
    return dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


def _build_model(feature_count):
    inputs = tf.keras.Input((IMAGE_SIZE, IMAGE_SIZE, 3), name="image")
    x = inputs
    for filters in (24, 32, 48):
        x = tf.keras.layers.SeparableConv2D(
            filters, 3, strides=2, padding="same", use_bias=False
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("swish")(x)
    x = tf.keras.layers.SeparableConv2D(
        64, 3, padding="same", use_bias=False
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("swish")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    outputs = tf.keras.layers.Dense(
        feature_count, activation="sigmoid", name="feature_probabilities"
    )(x)
    model = tf.keras.Model(inputs, outputs, name="raabin_feature_cnn")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=7e-4),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.BinaryAccuracy(name="binary_accuracy")],
    )
    return model


def _feature_metrics(targets, probabilities, feature_names):
    predictions = probabilities >= 0.5
    precision, recall, f1, _ = precision_recall_fscore_support(
        targets, predictions, average=None, zero_division=0
    )
    metrics = {}
    defined_f1 = []
    for index, name in enumerate(feature_names):
        positive_support = int(np.sum(targets[:, index] == 1))
        has_positive = positive_support > 0
        feature_f1 = float(f1[index]) if has_positive else None
        if feature_f1 is not None:
            defined_f1.append(feature_f1)
        metrics[name] = {
            "accuracy": float(
                accuracy_score(targets[:, index], predictions[:, index])
            ),
            "precision": float(precision[index]) if has_positive else None,
            "recall": float(recall[index]) if has_positive else None,
            "f1": feature_f1,
            "positive_weak_target_count": positive_support,
            "note": (
                None
                if has_positive
                else "No positive class-derived training targets exist."
            ),
        }
    return metrics, float(np.mean(defined_f1)) if defined_f1 else None


def _write_validation_csv(rows, targets, probabilities, feature_names, output):
    records = pd.DataFrame(
        {
            "image_filename": [Path(path).name for path in rows["image_path"]],
            "image_path": rows["image_path"],
            "actual_wbc_class": rows["label"],
            "source_group_id": rows["group_id"],
            "preprocessing_qc": rows["qc_flag"],
            "prediction_partition": "validation",
        }
    )
    records["feature_vector"] = [
        json.dumps([float(value) for value in vector], separators=(",", ":"))
        for vector in probabilities
    ]
    for index, feature in enumerate(feature_names):
        records[f"{feature}_weak_target"] = targets[:, index].astype(np.int32)
        records[f"{feature}_probability"] = probabilities[:, index]
        records[f"{feature}_prediction"] = (
            probabilities[:, index] >= 0.5
        ).astype(np.int32)
    output.parent.mkdir(parents=True, exist_ok=True)
    records.to_csv(output, index=False)
    return records


def train(manifest_path, processed_manifest_path, epochs=MAX_EPOCHS, seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except RuntimeError:
        pass

    feature_names, class_targets = _load_targets()
    rows, skipped_rows = _load_training_rows(
        _resolve_path(manifest_path), _resolve_path(processed_manifest_path)
    )
    labels = rows["label"].to_numpy()
    groups = rows["group_id"].astype(str).to_numpy()
    targets = np.asarray([class_targets[label] for label in labels], dtype=np.float32)

    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    train_indices, validation_indices = next(
        splitter.split(np.zeros(len(rows)), labels, groups)
    )
    train_groups = set(groups[train_indices])
    validation_groups = set(groups[validation_indices])
    if train_groups.intersection(validation_groups):
        raise RuntimeError("A source group appears in both training and validation.")

    train_labels = labels[train_indices]
    class_counts = pd.Series(train_labels).value_counts()
    missing_classes = sorted(set(CLASSES).difference(class_counts.index))
    if missing_classes:
        raise ValueError(f"Training partition is missing classes: {missing_classes}")
    class_weights = {
        label: len(train_indices) / (len(CLASSES) * int(class_counts[label]))
        for label in CLASSES
    }
    sample_weights = np.asarray(
        [class_weights[label] for label in train_labels], dtype=np.float32
    )
    paths = rows["image_path"].to_numpy(dtype=str)
    train_data = _make_dataset(
        paths[train_indices],
        targets[train_indices],
        sample_weights,
        shuffle=True,
        seed=seed,
    )
    validation_data = _make_dataset(
        paths[validation_indices], targets[validation_indices]
    )

    model_path = DEFAULT_MODEL
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model = _build_model(len(feature_names))
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            model_path, monitor="val_loss", mode="min", save_best_only=True
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=5,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5
        ),
    ]
    history = model.fit(
        train_data,
        validation_data=validation_data,
        epochs=epochs,
        callbacks=callbacks,
        shuffle=False,
        verbose=2,
    )
    pd.DataFrame(history.history).rename_axis("epoch").to_csv(
        DEFAULT_HISTORY_CSV, index=True
    )

    validation_probabilities = model.predict(validation_data, verbose=0)
    validation_rows = rows.iloc[validation_indices].reset_index(drop=True)
    validation_targets = targets[validation_indices].astype(np.int32)
    validation_csv = _write_validation_csv(
        validation_rows,
        validation_targets,
        validation_probabilities,
        feature_names,
        DEFAULT_VALIDATION_CSV,
    )
    feature_metrics, macro_f1 = _feature_metrics(
        validation_targets, validation_probabilities, feature_names
    )
    report = {
        "dataset": "raabin",
        "split_used": "train only",
        "target_source": (
            "Class-derived weak supervision; not per-image human morphology labels."
        ),
        "important_limitations": [
            "Feature scores estimate the class-derived targets and do not establish "
            "accurate visual morphology recognition.",
            "dense_blue_granules has no positive class-derived targets and cannot "
            "be learned as a positive feature from this mapping.",
            "No RAABIN test split was used for training or evaluated here.",
        ],
        "image_size": [IMAGE_SIZE, IMAGE_SIZE],
        "train_images": int(len(train_indices)),
        "validation_images": int(len(validation_indices)),
        "skipped_manifest_rows_without_processed_images": skipped_rows,
        "training_class_counts": {
            label: int(class_counts[label]) for label in CLASSES
        },
        "class_weights": class_weights,
        "group_overlap": False,
        "seed": seed,
        "epochs_completed": len(history.history["loss"]),
        "best_epoch": int(np.argmin(history.history["val_loss"])) + 1,
        "validation_loss_at_best_epoch": float(min(history.history["val_loss"])),
        "validation_macro_f1_on_weak_targets": macro_f1,
        "validation_metrics_by_feature_on_weak_targets": feature_metrics,
        "model_path": str(model_path.resolve()),
        "validation_csv": str(DEFAULT_VALIDATION_CSV.resolve()),
        "history_csv": str(DEFAULT_HISTORY_CSV.resolve()),
    }
    DEFAULT_METRICS_JSON.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_METRICS_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved CNN model: {model_path}")
    print(f"Saved held-out validation CSV: {DEFAULT_VALIDATION_CSV}")
    print(f"Saved validation report: {DEFAULT_METRICS_JSON}")
    print(f"Saved training history: {DEFAULT_HISTORY_CSV}")
    print(
        f"Validation macro F1 on weak targets: "
        f"{macro_f1 if macro_f1 is not None else 'n/a'}"
    )
    print(
        f"Rows: {len(validation_csv)} validation predictions; "
        f"skipped {skipped_rows} manifest rows without processed images."
    )


def predict(image_paths, model_path, output_path):
    model = tf.keras.models.load_model(model_path, compile=False)
    feature_names, _ = _load_targets()
    resolved_paths = [_resolve_path(path) for path in image_paths]
    missing_paths = [str(path) for path in resolved_paths if not path.is_file()]
    if missing_paths:
        raise FileNotFoundError(f"Input images not found: {missing_paths}")
    dataset = _make_dataset(np.asarray([str(path) for path in resolved_paths]))
    probabilities = model.predict(dataset, verbose=0)
    records = {
        "image_filename": [path.name for path in resolved_paths],
        "image_path": [str(path) for path in resolved_paths],
    }
    records["feature_vector"] = [
        json.dumps([float(value) for value in vector], separators=(",", ":"))
        for vector in probabilities
    ]
    for index, feature in enumerate(feature_names):
        records[f"{feature}_probability"] = probabilities[:, index]
        records[f"{feature}_prediction"] = (
            probabilities[:, index] >= 0.5
        ).astype(np.int32)
    output = _resolve_path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(output, index=False)
    print(f"Saved predictions: {output}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Train a RAABIN-only CNN for class-derived feature estimates, or "
            "run the saved model on processed PNG images."
        )
    )
    commands = parser.add_subparsers(dest="command", required=True)
    train_parser = commands.add_parser("train", help="Train and evaluate on RAABIN train.")
    train_parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    train_parser.add_argument(
        "--processed-manifest",
        type=Path,
        default=DEFAULT_PROCESSED_MANIFEST,
    )
    train_parser.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    train_parser.add_argument("--seed", type=int, default=SEED)
    predict_parser = commands.add_parser(
        "predict", help="Load the saved CNN and predict processed PNG images."
    )
    predict_parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    predict_parser.add_argument(
        "--images", type=Path, nargs="+", required=True
    )
    predict_parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "raabin_feature_predictions.csv",
    )
    args = parser.parse_args()
    if args.command == "train":
        if args.epochs < 1:
            parser.error("--epochs must be positive.")
        train(args.manifest, args.processed_manifest, args.epochs, args.seed)
    else:
        predict(args.images, args.model, args.output)


if __name__ == "__main__":
    main()
