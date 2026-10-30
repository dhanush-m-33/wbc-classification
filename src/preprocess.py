"""
WBC preprocessing: nucleus-guided cell detection (adaptive thresholds).

Run from the project root.

Training data (Raabin train): detect, keep only good crops.
    python src/preprocess.py --only raabin_train --workers 4

Test data: never drop images. Every image gets a crop; hard ones fall back to
the whole image. Evaluate from data/manifests/eval_<name>.csv.
    python src/preprocess.py --only raabin_testA --fallback full --review-policy full --workers 4
    python src/preprocess.py --only raabin_testB --fallback full --review-policy full --workers 4
    python src/preprocess.py --only lisc         --fallback full --review-policy full --workers 4

Inputs that are already tight single-cell crops (BCCD): skip detection.
    python src/preprocess.py --only bccd --assume-cropped --workers 4

Quick check on a sample (takes --limit rows PER CLASS):
    python src/preprocess.py --only raabin_train --limit 20 --workers 4

Each manifest must contain: path, dataset, split, label.

Output:
    data/processed/<dataset>/<split>/<label>/*.png
    data/processed/qc_<manifest>.csv           every image + QC columns
    data/processed/preview_<manifest>.png
    data/manifests/processed_<manifest>.csv    accepted rows only (use for TRAINING)
    data/manifests/eval_<manifest>.csv         every written crop (use for TESTING)

QC flags:
    ok                              cell extent refined from colour foreground (accepted)
    nucleus_scaled                  crop sized from nucleus cluster (accepted)
    assumed_crop                    --assume-cropped: whole image used, no detection (accepted)
    cell_touches_image_edge_review  cell touches the border
    off_centre_nucleus_review       selected nucleus far from image centre
    edge_padding_review             crop needed padding
    near_full_image_crop_review     crop covers >=90% of the image
    large_cell_box_review           box covers >60% of the image
    full_image_review               --review-policy full replaced a flagged crop by the whole image
    full_image_fallback_review      no nucleus found; --fallback full used the whole image
Review flags are excluded from processed_<manifest>.csv unless --accept-review.
Original images are never modified.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

MANIFESTS = ["raabin_train", "raabin_testA", "raabin_testB", "lisc", "bccd"]

# ---- tunables ---------------------------------------------------------------
NUC_CHROMA_MIN = 35      # (R+B)/2 - G must exceed this for a nucleus pixel
NUC_MIN_FRAC = 0.0008    # min nucleus-cluster area / image area
NUC_MAX_FRAC = 0.15      # max nucleus-cluster area / image area
MIN_CELL_RATIO = 1.15    # refined cell box must exceed nucleus box by this
MAX_CELL_RATIO = 2.0     # cell box is clipped to this x nucleus box (blocks RBC merging)
FALLBACK_RATIO = 1.6     # fallback square side = this x nucleus box
FALLBACK_MIN_FRAC = 0.38 # fallback side limits, relative to min(sh, sw)
FALLBACK_MAX_FRAC = 0.60
OFF_CENTRE_LIMIT = 0.45  # normalised distance beyond which a nucleus is flagged
NUC_MEAN_CHROMA_MIN = 0

ACCEPTED_DEFAULT = {"ok", "nucleus_scaled", "assumed_crop"}
ALL_REVIEW = {
    "cell_touches_image_edge_review",
    "off_centre_nucleus_review",
    "edge_padding_review",
    "near_full_image_crop_review",
    "large_cell_box_review",
    "full_image_review",
    "full_image_fallback_review",
}
# Flags where --review-policy full swaps the crop for the whole image.
REPLACE_FLAGS = {
    "cell_touches_image_edge_review",
    "off_centre_nucleus_review",
    "large_cell_box_review",
    "near_full_image_crop_review",
}
NO_NUCLEUS_FLAGS = {"no_plausible_nucleus", "full_image_fallback_review"}
FULL_IMAGE_FLAGS = {"full_image_review", "full_image_fallback_review"}


def _odd(v, lo, hi):
    v = max(lo, min(hi, int(round(v))))
    if v % 2 == 0:
        v = v + 1 if v + 1 <= hi else v - 1
    return v


def _ellipse(k):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def _scale_for_detection(rgb, max_dim=800):
    h, w = rgb.shape[:2]
    scale = min(1.0, float(max_dim) / max(h, w))
    if scale < 1.0:
        small = cv2.resize(
            rgb,
            (max(8, round(w * scale)), max(8, round(h * scale))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        small = rgb.copy()
    return small, scale


def border_colour(rgb):
    """Estimate background colour from the image borders."""
    h, w = rgb.shape[:2]
    t = max(1, min(5, h // 20 or 1, w // 20 or 1))
    ring = np.concatenate(
        [
            rgb[:t].reshape(-1, 3),
            rgb[-t:].reshape(-1, 3),
            rgb[:, :t].reshape(-1, 3),
            rgb[:, -t:].reshape(-1, 3),
        ],
        axis=0,
    )
    return np.median(ring, axis=0).astype(np.uint8)


# -----------------------------------------------------------------------------
# 1. Nucleus detection
# -----------------------------------------------------------------------------
def _pick_nucleus_cluster(mask, chroma, sh, sw, centre_weight):
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, _ellipse(_odd(0.01 * min(sh, sw), 3, 9))
    )
    if not mask.any():
        return None

    # Merge lobes of the same nucleus (segmented nuclei have separate lobes).
    gap = _odd(0.025 * min(sh, sw), 5, 41)
    merged = cv2.dilate(mask, _ellipse(gap))
    n, labels = cv2.connectedComponents(merged)

    img_area = float(sh * sw)
    half_diag = np.hypot(sw, sh) / 2.0
    best = None

    for lab in range(1, n):
        pix = (labels == lab) & (mask > 0)
        area = int(pix.sum())
        frac = area / img_area
        if frac < NUC_MIN_FRAC or frac > NUC_MAX_FRAC:
            continue

        ys, xs = np.nonzero(pix)
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        bw, bh = x1 - x0, y1 - y0
        if bw < 5 or bh < 5:
            continue

        if bw > 0.85 * sw or bh > 0.85 * sh:          # stain / scratch spanning the image
            continue
        if max(bw, bh) / min(bw, bh) > 3.0:           # thin streak, not a nucleus
            continue

        dist = np.hypot(xs.mean() - sw / 2.0, ys.mean() - sh / 2.0) / half_diag

        mean_chroma = float(chroma[pix].mean())
        if mean_chroma < NUC_MEAN_CHROMA_MIN:
            continue

        score = np.sqrt(frac) - centre_weight * dist

        if best is None or score > best["score"]:
            best = {
                "score": float(score),
                "box": (x0, y0, bw, bh),
                "fraction": float(frac),
                "distance": float(dist),
                "chroma": mean_chroma,
            }
    return best


def detect_nucleus(small, centre_weight):
    sh, sw = small.shape[:2]
    blur = cv2.GaussianBlur(small, (5, 5), 0)
    g8 = blur[:, :, 1]

    chroma = (
        (blur[:, :, 0].astype(np.int16) + blur[:, :, 2].astype(np.int16)) // 2
        - g8.astype(np.int16)
    )

    t1, _ = cv2.threshold(g8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dark_vals = g8[g8 < t1]

    thresholds = []
    if dark_vals.size >= 50:
        t2, _ = cv2.threshold(
            dark_vals.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        thresholds.append(t2)   # darkest class: nucleus
    thresholds.append(t1)       # looser retry

    best = None
    for thr in thresholds:
        mask = ((g8 <= thr) & (chroma >= NUC_CHROMA_MIN)).astype(np.uint8) * 255
        found = _pick_nucleus_cluster(mask, chroma, sh, sw, centre_weight)
        if found is not None and (best is None or found["score"] > best["score"]):
            best = found
    return best


# -----------------------------------------------------------------------------
# 2. Cell extent from background-distance foreground
# -----------------------------------------------------------------------------
def refine_cell_box(small, nuc, bg):
    sh, sw = small.shape[:2]
    blur = cv2.GaussianBlur(small, (5, 5), 0)
    diff = np.linalg.norm(blur.astype(np.float32) - bg.astype(np.float32), axis=2)
    d8 = np.clip(diff, 0, 255).astype(np.uint8)

    t, _ = cv2.threshold(d8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    fg = (d8 > t).astype(np.uint8) * 255

    k = _ellipse(_odd(0.02 * min(sh, sw), 3, 15))
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k)

    n, labels = cv2.connectedComponents(fg)
    nx, ny, nw, nh = nuc["box"]

    sub = labels[ny:ny + nh, nx:nx + nw]
    sub = sub[sub > 0]
    if sub.size == 0:
        return None
    lab = int(np.bincount(sub).argmax())

    ys, xs = np.nonzero(labels == lab)
    x0, x1 = float(xs.min()), float(xs.max() + 1)
    y0, y1 = float(ys.min()), float(ys.max() + 1)

    # Clip so touching RBCs / neighbours cannot inflate the box.
    ncx, ncy = nx + nw / 2.0, ny + nh / 2.0
    min_dim = float(min(sh, sw))
    half = min(MAX_CELL_RATIO * max(nw, nh), FALLBACK_MAX_FRAC * min_dim) / 2.0
    x0, x1 = max(x0, ncx - half), min(x1, ncx + half)
    y0, y1 = max(y0, ncy - half), min(y1, ncy + half)

    # Always contain the whole nucleus.
    x0, y0 = min(x0, nx), min(y0, ny)
    x1, y1 = max(x1, nx + nw), max(y1, ny + nh)

    if (x1 - x0) < MIN_CELL_RATIO * nw or (y1 - y0) < MIN_CELL_RATIO * nh:
        return None

    # Floor: very small boxes get context instead of an extreme zoom.
    floor = FALLBACK_MIN_FRAC * min_dim
    if max(x1 - x0, y1 - y0) < floor:
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        x0, x1 = cx - floor / 2.0, cx + floor / 2.0
        y0, y1 = cy - floor / 2.0, cy + floor / 2.0

    return x0, y0, x1, y1


def find_cell_box(rgb, max_dim=800, centre_weight=0.9):
    """
    Returns (box, meta); box is (x0, y0, x1, y1) in original coordinates or None.
    """
    h, w = rgb.shape[:2]
    small, scale = _scale_for_detection(rgb, max_dim)
    sh, sw = small.shape[:2]

    meta = {
        "detected": False,
        "detector_used": "none",
        "mask_fraction": 0.0,
        "component_area_fraction": 0.0,
        "box_area_fraction": 0.0,
        "component_centre_distance": np.nan,
        "qc_flag": "no_plausible_nucleus",
        "nuc_x": np.nan, "nuc_y": np.nan, "nuc_w": np.nan, "nuc_h": np.nan,
        "nuc_chroma": np.nan,
    }

    nuc = detect_nucleus(small, centre_weight)
    if nuc is None:
        return None, meta

    nx, ny, nw, nh = nuc["box"]
    meta.update(
        component_area_fraction=nuc["fraction"],
        component_centre_distance=nuc["distance"],
        nuc_x=nx / scale, nuc_y=ny / scale, nuc_w=nw / scale, nuc_h=nh / scale,
    )

    bg = border_colour(small)
    box = refine_cell_box(small, nuc, bg)

    if box is not None:
        x0, y0, x1, y1 = box
        touches = x0 <= 1 or y0 <= 1 or x1 >= sw - 1 or y1 >= sh - 1
        flag = "cell_touches_image_edge_review" if touches else "ok"
        detector = "nucleus_guided_foreground"
    else:
        ncx, ncy = nx + nw / 2.0, ny + nh / 2.0
        side = FALLBACK_RATIO * max(nw, nh)
        side = float(
            np.clip(
                side,
                FALLBACK_MIN_FRAC * min(sh, sw),
                FALLBACK_MAX_FRAC * min(sh, sw),
            )
        )
        side = max(side, float(max(nw, nh)))
        x0, y0 = ncx - side / 2.0, ncy - side / 2.0
        x1, y1 = ncx + side / 2.0, ncy + side / 2.0
        flag = "nucleus_scaled"
        detector = "nucleus_scaled_estimate"

    if nuc["distance"] > OFF_CENTRE_LIMIT and flag in ("ok", "nucleus_scaled"):
        flag = "off_centre_nucleus_review"

    # Back to original coordinates (box may extend past the image; the window
    # step shifts it back inside).
    box_orig = (x0 / scale, y0 / scale, x1 / scale, y1 / scale)
    box_orig = (
        max(0.0, box_orig[0]), max(0.0, box_orig[1]),
        min(float(w), box_orig[2]), min(float(h), box_orig[3]),
    )

    meta.update(
        detected=True,
        detector_used=detector,
        nuc_chroma=nuc["chroma"],
        box_area_fraction=float(
            (box_orig[2] - box_orig[0]) * (box_orig[3] - box_orig[1])
            / max(1.0, float(h * w))
        ),
        qc_flag=flag,
    )
    return box_orig, meta


# -----------------------------------------------------------------------------
# 3. Crop window
# -----------------------------------------------------------------------------
def make_window(rgb, box, margin, min_side_px, max_side_fraction):
    """Square window around the box, shifted (not cut) to stay inside the image."""
    h, w = rgb.shape[:2]
    x0, y0, x1, y1 = box

    bw = max(1.0, x1 - x0)
    bh = max(1.0, y1 - y0)
    detected_side = max(bw, bh)

    side = max(detected_side * (1.0 + 2.0 * margin), float(min_side_px))
    side = min(side, float(max(h, w)) * max_side_fraction)
    side = max(side, detected_side)          # never cut the cell
    if detected_side <= min(h, w):
        side = min(side, float(min(h, w)))   # avoid padding when the cell fits

    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    x, y = cx - side / 2.0, cy - side / 2.0

    if side <= w:
        x = min(max(0.0, x), float(w) - side)
    if side <= h:
        y = min(max(0.0, y), float(h) - side)

    box_fraction = (bw * bh) / max(1.0, float(h * w))
    return float(x), float(y), float(side), float(box_fraction)


def crop_resize(rgb, win, size):
    x, y, side = win
    n = max(1, int(round(side)))
    ix, iy = int(round(x)), int(round(y))

    canvas = np.empty((n, n, 3), dtype=np.uint8)
    canvas[:] = border_colour(rgb)

    sx0, sy0 = max(0, ix), max(0, iy)
    sx1, sy1 = min(rgb.shape[1], ix + n), min(rgb.shape[0], iy + n)

    if sx1 > sx0 and sy1 > sy0:
        dx, dy = sx0 - ix, sy0 - iy
        canvas[dy:dy + (sy1 - sy0), dx:dx + (sx1 - sx0)] = rgb[sy0:sy1, sx0:sx1]

    interp = cv2.INTER_AREA if n >= size else cv2.INTER_CUBIC
    return cv2.resize(canvas, (size, size), interpolation=interp)


def process_one(job):
    raw_path, out_path, p = job

    try:
        with Image.open(raw_path) as im:
            rgb = np.asarray(im.convert("RGB"))

        h, w = rgb.shape[:2]
        assume_cropped = bool(p.get("assume_cropped"))

        if assume_cropped:
            # Input is already a tight single-cell crop: no detection.
            box = (0.0, 0.0, float(w), float(h))
            meta = {
                "detector_used": "assume_cropped",
                "qc_flag": "assumed_crop",
                "component_area_fraction": 0.0,
                "component_centre_distance": float("nan"),
            }
        else:
            box, meta = find_cell_box(rgb, centre_weight=p["centre_weight"])

        nuc_cols = {k: float(meta.get(k, np.nan))
                    for k in ("nuc_x", "nuc_y", "nuc_w", "nuc_h", "nuc_chroma")}

        if box is None:
            if p["fallback"] == "skip":
                return {
                    "ok": False, "seg_ok": False,
                    "detector_used": meta["detector_used"],
                    "qc_flag": meta["qc_flag"],
                    "err": "No reliable nucleus detected",
                    **nuc_cols,
                }
            side = float(max(h, w))
            win = ((w - side) / 2.0, (h - side) / 2.0, side)
            detected, win_fraction, box_fraction = False, 1.0, 0.0
            flag = "full_image_fallback_review"
        else:
            x, y, side, box_fraction = make_window(
                rgb, box, p["margin"], p["min_side"], p["max_fraction"]
            )
            win = (x, y, side)
            detected = True
            win_fraction = side / max(h, w)
            flag = meta["qc_flag"]

            if not assume_cropped:
                if x < 0 or y < 0 or x + side > w or y + side > h:
                    flag = "edge_padding_review"
                if win_fraction >= 0.90:
                    flag = "near_full_image_crop_review"
                if box_fraction > 0.60:
                    flag = "large_cell_box_review"

                # Test-set policy: a flagged (probably wrong) crop is replaced
                # by the whole centred image instead of being dropped.
                if p.get("review_full") and (
                    flag in REPLACE_FLAGS or meta["qc_flag"] in REPLACE_FLAGS
                ):
                    side = float(max(h, w))
                    win = ((w - side) / 2.0, (h - side) / 2.0, side)
                    win_fraction = 1.0
                    flag = "full_image_review"

        output = crop_resize(rgb, win, p["size"])
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(output).save(out_path)

        means = output.reshape(-1, 3).mean(axis=0)
        return {
            "ok": True,
            "seg_ok": bool(detected),
            "detector_used": meta["detector_used"],
            "qc_flag": flag,
            "win_x": float(win[0]), "win_y": float(win[1]),
            "win_side": float(win[2]), "win_frac": float(win_fraction),
            "box_area_fraction": float(box_fraction),
            "nucleus_area_fraction": float(meta.get("component_area_fraction", 0.0)),
            "nucleus_centre_distance": float(
                meta.get("component_centre_distance", np.nan)
            ),
            **nuc_cols,
            "mean_r": float(means[0]), "mean_g": float(means[1]),
            "mean_b": float(means[2]),
            "raw_h": int(h), "raw_w": int(w),
            "err": "",
        }

    except Exception as exc:
        return {
            "ok": False, "seg_ok": False, "detector_used": "none",
            "qc_flag": "processing_error", "err": repr(exc),
        }


# -----------------------------------------------------------------------------
# Preview
# -----------------------------------------------------------------------------
def make_preview(df, root, name, n=12):
    """Top row: raw + crop window (red/yellow) + nucleus box (green). Bottom: crop."""
    valid = df[df["ok"].fillna(False)].copy()
    if valid.empty:
        print(f"  No successful images to preview for {name}.")
        return

    ok_flags = ACCEPTED_DEFAULT
    flagged = valid[~valid["qc_flag"].isin(ok_flags)]
    normal = valid[valid["qc_flag"].isin(ok_flags)]

    pieces = []
    if len(flagged):
        pieces.append(flagged.sample(min(len(flagged), max(1, n // 2)), random_state=7))
    remaining = n - sum(len(piece) for piece in pieces)
    if remaining > 0 and len(normal):
        pieces.append(normal.sample(min(len(normal), remaining), random_state=11))
    if not pieces:
        return

    subset = pd.concat(pieces).head(n)
    tile = 180
    sheet = Image.new("RGB", (len(subset) * tile, 2 * tile), (20, 24, 34))
    draw = ImageDraw.Draw(sheet)

    for index, row in enumerate(subset.itertuples(index=False)):
        try:
            raw_path = Path(str(row.raw_path))
            if not raw_path.is_absolute():
                raw_path = root / raw_path

            with Image.open(raw_path) as im:
                raw = im.convert("RGB")

            rw, rh = raw.size
            s = tile / max(rw, rh)
            sheet.paste(
                raw.resize((max(1, round(rw * s)), max(1, round(rh * s))),
                           Image.Resampling.LANCZOS),
                (index * tile, 0),
            )

            ox = index * tile
            x, y, side = float(row.win_x), float(row.win_y), float(row.win_side)
            draw.rectangle(
                [ox + x * s, y * s, ox + (x + side) * s, (y + side) * s],
                outline=(255, 55, 55) if bool(row.seg_ok) else (255, 190, 0),
                width=2,
            )

            if not np.isnan(row.nuc_x):
                draw.rectangle(
                    [ox + row.nuc_x * s, row.nuc_y * s,
                     ox + (row.nuc_x + row.nuc_w) * s, (row.nuc_y + row.nuc_h) * s],
                    outline=(60, 255, 90), width=1,
                )

            draw.text((ox + 3, 3), str(row.qc_flag)[:25], fill=(255, 255, 255))

            with Image.open(root / str(row.path)) as im:
                processed = im.convert("RGB")
            sheet.paste(
                processed.resize((tile, tile), Image.Resampling.LANCZOS),
                (ox, tile),
            )
        except Exception as exc:
            print(f"  Preview warning ({name}, row {index}): {exc}")

    out_dir = root / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    preview_path = out_dir / f"preview_{name}.png"
    sheet.save(preview_path)
    print(f"  Preview saved: {preview_path}")


def resolve_raw_path(root, raw_path):
    path = Path(str(raw_path))
    return path if path.is_absolute() else (root / path).resolve()


def main():
    parser = argparse.ArgumentParser(description="Preprocess WBC images and write QC reports.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--size", type=int, default=64, help="Output crop size")
    parser.add_argument("--margin", type=float, default=0.08)
    parser.add_argument("--min-side-px", type=int, default=32)
    parser.add_argument("--max-side-fraction", type=float, default=1.0)
    parser.add_argument("--fallback", choices=["full", "skip"], default="skip",
                        help="No nucleus found: skip the image, or use the whole image")
    parser.add_argument("--centre-weight", type=float, default=0.9,
                        help="Centre prior for choosing the nucleus (0 for multi-cell images)")
    parser.add_argument("--assume-cropped", action="store_true",
                        help="Inputs are already tight cell crops: use the whole image, no detection")
    parser.add_argument("--review-policy", choices=["keep", "full"], default="keep",
                        help="keep: leave flagged crops as they are; "
                             "full: replace flagged crops by the whole image (for test sets)")
    parser.add_argument("--accept-review", action="store_true",
                        help="Also put *_review rows in the processed manifest")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--only", default=None)
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N rows of each class")
    args = parser.parse_args()

    if args.size < 8 or args.margin < 0 or args.min_side_px < 8 or args.workers < 1:
        parser.error("Invalid size, margin, minimum side, or worker count.")
    if not 0.25 <= args.max_side_fraction <= 1.0:
        parser.error("--max-side-fraction must be between 0.25 and 1.0")

    params = {
        "size": args.size, "margin": args.margin, "min_side": args.min_side_px,
        "max_fraction": args.max_side_fraction, "fallback": args.fallback,
        "centre_weight": args.centre_weight,
        "assume_cropped": args.assume_cropped,
        "review_full": args.review_policy == "full",
    }
    accepted_flags = set(ACCEPTED_DEFAULT)
    if args.accept_review:
        accepted_flags |= ALL_REVIEW

    root = Path(args.root).resolve()
    names = [args.only] if args.only else MANIFESTS
    summary = []

    for name in names:
        manifest_path = root / "data" / "manifests" / f"{name}.csv"
        if not manifest_path.exists():
            print(f"[{name}] Manifest not found: {manifest_path}; skipping.")
            continue

        df = pd.read_csv(manifest_path)

        missing = {"path", "dataset", "split", "label"} - set(df.columns)
        if missing:
            print(f"[{name}] Missing columns {sorted(missing)}; skipping.")
            continue

        if args.limit is not None:
            df = df.groupby("label", group_keys=False).head(args.limit).copy()

        df = df.copy()
        df["raw_path"] = df["path"].astype(str)
        df["path"] = [
            (Path("data") / "processed" / str(r.dataset) / str(r.split)
             / str(r.label) / (Path(str(r.raw_path)).stem + ".png")).as_posix()
            for r in df.itertuples(index=False)
        ]

        jobs = [
            (str(resolve_raw_path(root, rp)), str(root / op), params)
            for rp, op in zip(df["raw_path"], df["path"])
        ]

        print(f"[{name}] Processing {len(jobs)} images ...", flush=True)

        if args.workers == 1:
            results = [process_one(j) for j in jobs]
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as ex:
                results = list(ex.map(process_one, jobs, chunksize=16))

        out = pd.concat([df.reset_index(drop=True), pd.DataFrame(results)], axis=1)

        out_dir = root / "data" / "processed"
        out_dir.mkdir(parents=True, exist_ok=True)
        qc_path = out_dir / f"qc_{name}.csv"
        out.to_csv(qc_path, index=False)

        ok_mask = out["ok"].fillna(False).astype(bool)
        errors = out[~ok_mask]

        # Training manifest: accepted rows only.
        accepted = out[
            ok_mask
            & out["seg_ok"].fillna(False).astype(bool)
            & out["qc_flag"].isin(accepted_flags)
        ].copy()

        manifest_dir = root / "data" / "manifests"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        cols = ["path"] + [c for c in accepted.columns if c not in {"path", "ok", "err"}]

        pm = manifest_dir / f"processed_{name}.csv"
        accepted[cols].to_csv(pm, index=False)

        # Evaluation manifest: every image that produced a crop, flagged or not.
        evaluation = out[ok_mask].copy()
        em = manifest_dir / f"eval_{name}.csv"
        evaluation[cols].to_csv(em, index=False)

        make_preview(out, root, name)

        total = len(out)
        good = int(ok_mask.sum())

        def pct(mask):
            return round(100.0 * mask.mean(), 1) if total else None

        summary.append({
            "manifest": name,
            "input_n": total,
            "output_files_n": good,
            "accepted_n": len(accepted),
            "no_nucleus_n": int(out["qc_flag"].isin(NO_NUCLEUS_FLAGS).sum()),
            "errors": int((out["qc_flag"] == "processing_error").sum()),
            "refined_pct": pct(out["qc_flag"].eq("ok")),
            "nucleus_scaled_pct": pct(out["qc_flag"].eq("nucleus_scaled")),
            "assumed_pct": pct(out["qc_flag"].eq("assumed_crop")),
            "full_image_pct": pct(out["qc_flag"].isin(FULL_IMAGE_FLAGS)),
            "review_pct": pct(out["qc_flag"].isin(ALL_REVIEW)),
            "median_crop_fraction": (
                round(float(out.loc[ok_mask, "win_frac"].median()), 3)
                if good else None
            ),
        })

        print(f"  Output images: {good}/{total}; accepted manifest rows: {len(accepted)}")
        print(f"  QC report: {qc_path}")
        print(f"  Training manifest: {pm}")
        print(f"  Evaluation manifest: {em}")
        if len(errors):
            print("  Sample failures:",
                  errors[["raw_path", "qc_flag", "err"]].head(3).to_dict("records"))

    if summary:
        sdf = pd.DataFrame(summary)
        sp = root / "data" / "processed" / "qc_summary.csv"
        sdf.to_csv(sp, index=False)
        print("\nQC summary:")
        print(sdf.to_string(index=False))
        print("\nReview: data/processed/preview_<manifest>.png and qc_<manifest>.csv")
        print("If you used --limit, rerun without --limit before training.")


if __name__ == "__main__":
    main()