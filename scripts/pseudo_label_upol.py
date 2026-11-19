"""Self-training data: pseudo-label UPOL TRAIN-split close-ups with the classical Daugman
segmenter and keep only results that pass strict automatic QC. Val/test UPOL images are
never touched, so the held-out visual grade stays independent.

python scripts/pseudo_label_upol.py   -> data/labels/upol_pseudo.csv + masks in data/cache/masks/upol_pseudo/
then python scripts/build_manifest.py adds them as dataset 'upol_pseudo' (train split only).
"""
from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np

from aeye_iris.segment import build
from aeye_iris.segment.classical import LIMBUS_ANGLES, PUPIL_ANGLES, _ido_search
from aeye_iris.preprocess import prepare, resize_max

OUT = Path("data/cache/masks/upol_pseudo")


def edge(gs, s, e, angles):
    R = e.r * s
    radii = np.arange(max(2.0, 0.9 * R), 1.1 * R + 1, 0.5, dtype=np.float32)
    return _ido_search(gs, np.array([[e.cx * s, e.cy * s]], np.float32), radii, angles, 1.0)[2]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    seg = build("daugman", channel="red", clahe=True, inpaint=True, rmax_frac=0.6)
    rows = [r for r in csv.DictReader(open("data/manifest.csv")) if r["dataset"] == "upol" and r["split"] == "train"]
    kept, out = 0, []
    for r in rows:
        img = cv2.imread(r["image"])
        h, w = img.shape[:2]
        res = seg(img)
        if not res.ok:
            continue
        g = prepare(img, "red", True, True)[0]
        gs, s = resize_max(g, 240)
        gs = cv2.GaussianBlur(gs, (0, 0), 1.0).astype(np.float32)
        ratio = res.pupil.r / res.iris.r
        conc = np.hypot(res.pupil.cx - res.iris.cx, res.pupil.cy - res.iris.cy) / res.iris.r
        frac = res.iris.r / min(h, w)
        el, ep = edge(gs, s, res.iris, LIMBUS_ANGLES), edge(gs, s, res.pupil, PUPIL_ANGLES)
        ok = 0.12 <= ratio <= 0.6 and conc < 0.12 and frac >= 0.40 and el > 8 and ep > 8
        if not ok:
            continue
        stem = Path(r["image"]).stem
        ip, pp = OUT / f"{stem}_iris.png", OUT / f"{stem}_pupil.png"
        cv2.imwrite(str(ip), (res.iris_mask > 0).astype(np.uint8) * 255)
        cv2.imwrite(str(pp), (res.pupil_mask > 0).astype(np.uint8) * 255)
        out.append({"id": f"upol_pseudo/{stem}", "image": r["image"], "iris_mask": str(ip), "pupil_mask": str(pp),
                    "subject": r["subject"], "edge_limbus": round(el, 1), "edge_pupil": round(ep, 1)})
        kept += 1
    with open("data/labels/upol_pseudo.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, list(out[0].keys()))
        wr.writeheader()
        wr.writerows(out)
    print(f"kept {kept}/{len(rows)} UPOL train images as pseudo-labels")


if __name__ == "__main__":
    main()
