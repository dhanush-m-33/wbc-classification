"""
Build the Raabin-WBC manifests (Train / TestA / TestB).

Run from the project root:   python src/build_manifest_raabin.py

Input : data/raw/raabin/  with Train, TestA, TestB folders, each holding class folders:
        Train/Neutrophil/20190526_162951_0.jpg
Output: data/manifests/raabin_train.csv, raabin_testA.csv, raabin_testB.csv
        data/manifests/raabin_preview.png   (6 random Train cells per class)
Columns: path, label, label_idx, dataset, split, group_id, width, height

group_id = the date_time part of the filename (20190526_162951). ASSUMPTION: this is the source
smear image, so all cells cut from one image share it. The report below tells you whether that
looks right (cells per group) and whether any group appears in more than one split (leakage).
"""
import hashlib
import random
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
RAW = ROOT / "data" / "raw" / "raabin"
OUT = ROOT / "data" / "manifests"
OUT.mkdir(parents=True, exist_ok=True)

# ---- shared label space (identical in every dataset script; never reorder) ----
CLASSES = ["eosinophil", "lymphocyte", "monocyte", "neutrophil"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
ALIASES = {"eosinophil": "eosinophil", "lymphocyte": "lymphocyte",
           "monocyte": "monocyte", "neutrophil": "neutrophil"}      # basophil deliberately absent
SPLIT_NAMES = {"train": "train", "testa": "testA", "test-a": "testA", "test_a": "testA",
               "testb": "testB", "test-b": "testB", "test_b": "testB"}
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
GROUP_RE = re.compile(r"^(\d{8}_\d{6})")


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find_split(rel_parts):
    for part in rel_parts[:-1]:
        s = SPLIT_NAMES.get(part.lower())
        if s:
            return s
    return None


def main():
    if not RAW.exists():
        sys.exit(f"Folder not found: {RAW}")
    other_files = Counter()
    dropped = Counter()
    bad = {"no_split": 0, "unreadable": [], "dup_within": 0, "dup_cross": []}
    rows, seen = [], {}

    # Train first, so a cross-split duplicate is removed from the TEST side, never from Train
    order = {"train": 0, "testA": 1, "testB": 2}
    files = sorted(RAW.rglob("*"), key=lambda p: (order.get(find_split(p.relative_to(RAW).parts) or "", 9), str(p)))
    for p in files:
        if not p.is_file():
            continue
        if p.suffix.lower() not in IMG_EXT:
            other_files[p.suffix.lower() or "(no extension)"] += 1       # e.g. .json/.csv label files
            continue
        rel = p.relative_to(RAW).parts
        split = find_split(rel)
        if split is None:
            bad["no_split"] += 1
            continue
        label = ALIASES.get(p.parent.name.strip().lower())
        if label is None:
            dropped[p.parent.name] += 1                                  # basophil etc.
            continue
        try:
            with Image.open(p) as im:
                im.load()
                w, h = im.size
        except Exception as e:
            bad["unreadable"].append(f"{p.name}: {e}")
            continue
        d = md5(p)
        if d in seen:
            first_split, first_name = seen[d]
            if first_split == split:
                bad["dup_within"] += 1
            else:
                bad["dup_cross"].append(f"{split}/{p.name} == {first_split}/{first_name}")
            continue
        seen[d] = (split, p.name)
        m = GROUP_RE.match(p.stem)
        rows.append(dict(path=p.relative_to(ROOT).as_posix(), label=label, label_idx=CLASS_TO_IDX[label],
                         dataset="raabin", split=split, group_id="raabin_" + (m.group(1) if m else p.stem),
                         width=w, height=h))

    df = pd.DataFrame(rows)
    if df.empty:
        sys.exit("No usable Raabin images found. Check the folder layout (Train/TestA/TestB -> class folders).")
    for split, g in df.groupby("split"):
        g.to_csv(OUT / f"raabin_{split}.csv", index=False)

    print(f"\nRaabin manifests written to {OUT} : {len(df)} cells in total")
    print("\nCells per split and class:")
    print(pd.crosstab(df.split, df.label).reindex(columns=CLASSES).fillna(0).astype(int).to_string())
    print("\nSource-image groups per split:", df.groupby("split").group_id.nunique().to_dict())
    per_group = df.groupby("group_id").size()
    print("Cells per group (min / median / max):", per_group.min(), int(per_group.median()), per_group.max())
    gs = {s: set(g.group_id) for s, g in df.groupby("split")}
    for a, b in [("train", "testA"), ("train", "testB"), ("testA", "testB")]:
        if a in gs and b in gs:
            print(f"Groups shared by {a} and {b}: {len(gs[a] & gs[b])}   <- should be 0 (else leakage)")
    print("\nImage sizes: most common =", df.groupby(["width", "height"]).size().sort_values(ascending=False).head(3).to_dict())
    print("Dropped (folder not one of the 4 classes):", dict(dropped))
    print("Non-image files found (look for label files!):", dict(other_files))
    print("Files outside Train/TestA/TestB:", bad["no_split"])
    print("Unreadable:", len(bad["unreadable"]), bad["unreadable"][:3])
    print("Duplicates inside a split:", bad["dup_within"])
    print("Duplicates ACROSS splits (removed from the test side):", len(bad["dup_cross"]), bad["dup_cross"][:3])

    random.seed(0)
    tile, per = 112, 6
    sheet = Image.new("RGB", (per * tile + 130, len(CLASSES) * (tile + 4)), (20, 24, 34))
    d = ImageDraw.Draw(sheet)
    tr = df[df.split == "train"]
    for r, c in enumerate(CLASSES):
        sub = tr[tr.label == c]
        d.text((8, r * (tile + 4) + tile // 2), c, fill=(255, 255, 255))
        for k, rel in enumerate(sub.sample(min(per, len(sub)), random_state=0).path):
            sheet.paste(Image.open(ROOT / rel).convert("RGB").resize((tile, tile)), (130 + k * tile, r * (tile + 4)))
    sheet.save(OUT / "raabin_preview.png")
    print(f"\nPreview written: {OUT / 'raabin_preview.png'}")


if __name__ == "__main__":
    main()