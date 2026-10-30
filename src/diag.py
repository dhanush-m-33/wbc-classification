import sys
sys.path.insert(0, "src")
import numpy as np, pandas as pd, cv2
from PIL import Image
import preprocess as P

name = sys.argv[1]
d = pd.read_csv(f"data/processed/qc_{name}.csv")
groups = [
    ("FAILED", d[d.qc_flag == "no_plausible_nucleus"].head(8)),
    ("OK", d[d.qc_flag.isin(["ok", "nucleus_scaled"])].head(8)),
]
for tag, g in groups:
    for p in g.raw_path:
        rgb = np.asarray(Image.open(p).convert("RGB"))
        small, _ = P._scale_for_detection(rgb)
        blur = cv2.GaussianBlur(small, (5, 5), 0)
        g8 = blur[:, :, 1]
        chroma = (blur[:, :, 0].astype(int) + blur[:, :, 2].astype(int)) // 2 - g8.astype(int)
        t1, _ = cv2.threshold(g8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        print(tag, rgb.shape[:2], "otsu", int(t1),
              "chroma p99", int(np.percentile(chroma, 99)),
              "median", int(np.percentile(chroma, 50)))