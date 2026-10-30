"""Build a group-separated, class-balanced morphology annotation worksheet.

Run from the project root:
    python src/build_feature_annotation_template.py

The worksheet includes no model predictions or class labels. Its training and
validation partitions are group-separated and class-stratified.
"""

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "manifests" / "raabin_train.csv"
DEFAULT_PROCESSED_MANIFEST = (
    PROJECT_ROOT / "data" / "manifests" / "processed_raabin_train.csv"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "results" / "raabin_feature_annotation_template.csv"
)
CLASSES = ("eosinophil", "lymphocyte", "monocyte", "neutrophil")
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


def _read_annotation_candidates(manifest_path, processed_manifest_path):
    manifest = pd.read_csv(manifest_path)
    processed = pd.read_csv(processed_manifest_path)

    required_manifest = {"path", "label", "dataset", "split", "group_id"}
    required_processed = {"raw_path", "path"}
    for name, frame, required in (
        ("training manifest", manifest, required_manifest),
        ("processed manifest", processed, required_processed),
    ):
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"{name} is missing required columns: {missing}")

    manifest["label"] = manifest["label"].astype(str).str.strip().str.lower()
    if not manifest["dataset"].astype(str).str.lower().eq("raabin").all():
        raise ValueError("The training manifest contains non-RAABIN rows.")
    if not manifest["split"].astype(str).str.lower().eq("train").all():
        raise ValueError("The training manifest must contain only the train split.")
    unknown_classes = sorted(set(manifest["label"]).difference(CLASSES))
    if unknown_classes:
        raise ValueError(f"Unsupported WBC classes in manifest: {unknown_classes}")

    rows = manifest.merge(
        processed[["raw_path", "path"]].rename(columns={"path": "processed_path"}),
        left_on="path",
        right_on="raw_path",
        how="left",
        validate="one_to_one",
    )
    rows = rows.loc[rows["processed_path"].notna()].copy()
    rows["image_path"] = rows["processed_path"].map(
        lambda value: PROJECT_ROOT / Path(value)
    )
    rows = rows.loc[rows["image_path"].map(Path.is_file)].copy()
    rows["image_filename"] = rows["image_path"].map(lambda path: path.name)

    if rows.empty:
        raise ValueError(
            "No accepted training images match the processed manifest."
        )
    return rows.reset_index(drop=True)


def build_annotation_template(
    manifest_path=DEFAULT_MANIFEST,
    processed_manifest_path=DEFAULT_PROCESSED_MANIFEST,
    output_path=DEFAULT_OUTPUT,
    train_per_class=100,
    validation_per_class=25,
    seed=42,
):
    if train_per_class < 1 or validation_per_class < 1:
        raise ValueError("Sample counts must be positive integers.")

    candidates = _read_annotation_candidates(manifest_path, processed_manifest_path)
    labels = candidates["label"].to_numpy()
    groups = candidates["group_id"].astype(str).to_numpy()
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    _, validation_indices = next(
        splitter.split(np.zeros(len(candidates)), labels, groups)
    )
    validation_groups = set(groups[validation_indices])
    candidates["annotation_partition"] = candidates["group_id"].astype(str).map(
        lambda group: "validation" if group in validation_groups else "train"
    )

    samples = []
    for partition, per_class, random_state in (
        ("train", train_per_class, seed),
        ("validation", validation_per_class, seed + 1),
    ):
        partition_rows = candidates.loc[
            candidates["annotation_partition"] == partition
        ]
        for label in CLASSES:
            class_rows = partition_rows.loc[partition_rows["label"] == label]
            if len(class_rows) < per_class:
                raise ValueError(
                    f"Cannot select {per_class} {label} images for the "
                    f"{partition} annotation partition; only {len(class_rows)} "
                    "are available."
                )
            samples.append(
                class_rows.sample(n=per_class, random_state=random_state)
            )

    selected = pd.concat(samples, ignore_index=True)
    selected = selected.sort_values(
        ["annotation_partition", "group_id", "image_filename"],
        kind="stable",
    ).reset_index(drop=True)

    # Keep the annotation sheet free of predicted values and target class labels.
    output_path = Path(output_path)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    image_dir = output_path.parent / f"{output_path.stem}_images"
    image_dir.mkdir(parents=True, exist_ok=True)
    annotation_ids = [
        f"RAA-{index:04d}" for index in range(1, len(selected) + 1)
    ]
    image_paths = []
    for annotation_id, source_path in zip(annotation_ids, selected["image_path"]):
        image_path = image_dir / f"{annotation_id}.png"
        shutil.copyfile(source_path, image_path)
        try:
            image_paths.append(str(image_path.relative_to(PROJECT_ROOT)))
        except ValueError:
            image_paths.append(str(image_path.resolve()))

    worksheet = pd.DataFrame(
        {
            "annotation_id": annotation_ids,
            "annotation_partition": selected["annotation_partition"],
            "source_group_id": selected["group_id"],
            "image_path": image_paths,
            **{f"human_{feature}": "" for feature in FEATURES},
            "review_notes": "",
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    worksheet.to_csv(output_path, index=False)

    print(f"Saved annotation worksheet: {output_path}")
    print(f"Rows: {len(worksheet)}")
    print("Partitions:")
    print(
        selected.groupby(["annotation_partition", "label"])
        .size()
        .unstack(fill_value=0)
        .to_string()
    )
    print("Enter 1 for present, 0 for absent, and leave blank if uncertain.")


def main():
    parser = argparse.ArgumentParser(
        description="Create a group-separated WBC morphology annotation worksheet."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--processed-manifest",
        type=Path,
        default=DEFAULT_PROCESSED_MANIFEST,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train-per-class", type=int, default=100)
    parser.add_argument("--validation-per-class", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    build_annotation_template(
        manifest_path=args.manifest,
        processed_manifest_path=args.processed_manifest,
        output_path=args.output,
        train_per_class=args.train_per_class,
        validation_per_class=args.validation_per_class,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
