"""
Build the LISC manifest (test-only dataset).

Run from the project root:   python src/build_manifest_lisc.py

Input : data/raw/lisc/  (files like eosi_1-1.bmp, Baso_1-1.bmp, lymp_2-1.bmp ...; subfolders are fine)
Output: data/manifests/lisc.csv           one row per usable cell
        data/manifests/lisc_preview.png   6 random images per class, to eyeball the label mapping
Columns: path, label, label_idx, dataset, split, group_id, width, height

Checks done: filename parses, image opens, exact-duplicate files, augmented folders skipped,
basophil dropped. Everything dropped is printed, never silently ignored.
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
RAW = ROOT / "data" / "raw" / "lisc"
OUT = ROOT / "data" / "manifests"
OUT.mkdir(parents=True, exist_ok=True)

# ---- shared label space (identical in every dataset script; never reorder) ----
CLASSES = ["eosinophil", "lymphocyte", "monocyte", "neutrophil"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
LISC_PREFIX = {"eosi": "eosinophil", "lymp": "lymphocyte", "mono": "monocyte", "neut": "neutrophil"}  # 'baso' dropped
IMG_EXT = {".bmp", ".png", ".jpg", ".jpeg"}
NAME_RE = re.compile(r"^([a-z]+)[_-](\d+)-(\d+)$")      # eosi_10-1  ->  prefix, 10, 1


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    if not RAW.exists():
        sys.exit(f"Folder not found: {RAW}")
    rows = []
    dropped = Counter()
    seen_hash = {}
    problems = {"unparsed": [], "unreadable": [], "duplicate": [], "augmented_skipped": 0}

    for p in sorted(RAW.rglob("*")):
        if p.suffix.lower() not in IMG_EXT:
            continue
        if any("aug" in part.lower() for part in p.relative_to(RAW).parts):
            problems["augmented_skipped"] += 1           # e.g. a Train_Aug folder: never use for testing
            continue
        m = NAME_RE.match(p.stem.lower())
        if not m:
            problems["unparsed"].append(p.name)
            continue
        prefix, n1, _ = m.groups()
        label = LISC_PREFIX.get(prefix)
        if label is None:
            dropped[prefix] += 1                         # baso (and anything unknown)
            continue
        try:
            with Image.open(p) as im:
                im.load()
                w, h = im.size
        except Exception as e:                           # corrupt / truncated file
            problems["unreadable"].append(f"{p.name}: {e}")
            continue
        digest = md5(p)
        if digest in seen_hash:
            problems["duplicate"].append(f"{p.name} == {seen_hash[digest]}")
            continue
        seen_hash[digest] = p.name
        rows.append(dict(path=p.relative_to(ROOT).as_posix(), label=label, label_idx=CLASS_TO_IDX[label],
                         dataset="lisc", split="test", group_id=f"lisc_{prefix}_{n1}", width=w, height=h))

    df = pd.DataFrame(rows)
    if df.empty:
        sys.exit("No usable LISC images found. Check the folder layout and filenames.")
    df.to_csv(OUT / "lisc.csv", index=False)

    print(f"\nLISC manifest: {len(df)} cells -> {OUT / 'lisc.csv'}")
    print("\nPer class:\n" + df.label.value_counts().reindex(CLASSES).to_string())
    print("\nImage sizes (width x height):\n" + df.groupby(["width", "height"]).size().to_string())
    print("\nDropped by class prefix (not one of the 4 classes):", dict(dropped))
    print("Skipped as augmented:", problems["augmented_skipped"])
    for k in ("unparsed", "unreadable", "duplicate"):
        print(f"{k}: {len(problems[k])}", problems[k][:5])

    # ---- preview sheet: eyeball that each label really looks like its class ----
    random.seed(0)
    tile, per = 112, 6
    sheet = Image.new("RGB", (per * tile + 130, len(CLASSES) * (tile + 4)), (20, 24, 34))
    d = ImageDraw.Draw(sheet)
    for r, c in enumerate(CLASSES):
        sub = df[df.label == c]
        pick = sub.sample(min(per, len(sub)), random_state=0)
        d.text((8, r * (tile + 4) + tile // 2), c, fill=(255, 255, 255))
        for k, rel in enumerate(pick.path):
            im = Image.open(ROOT / rel).convert("RGB").resize((tile, tile))
            sheet.paste(im, (130 + k * tile, r * (tile + 4)))
    sheet.save(OUT / "lisc_preview.png")
    print(f"\nPreview written: {OUT / 'lisc_preview.png'}  (open it and check each row looks like its class)")


if __name__ == "__main__":
    main()