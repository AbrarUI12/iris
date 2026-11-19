"""Render segmentation overlays on a fixed sample of real visible-light images (UPOL,
no ground-truth masks) for visual pass/fail grading.

python -m bench.visual_sheet --configs "daugman|channel=red" "onnx|model=seg_small_224.onnx" --n 24
Grades are recorded in results/visual/grades.csv (tile -> pass/fail per config).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from aeye_iris.segment import build
from train.dataset import read_manifest


def parse(tag: str):
    parts = tag.split("|")
    kw = {}
    for p in parts[1:]:
        k, v = p.split("=", 1)
        if v in ("True", "False", "true", "false"):
            v = v.lower() == "true"
        else:
            try:
                v = float(v) if "." in v else int(v)
            except ValueError:
                pass
        kw[k] = v
    return parts[0], kw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--dataset", default="upol")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--out", default="results/visual")
    a = ap.parse_args()
    rows = read_manifest(a.manifest, "test", [a.dataset])
    rng = np.random.default_rng(0)
    rows = [rows[i] for i in sorted(rng.choice(len(rows), min(a.n, len(rows)), replace=False))]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for ci, tag in enumerate(a.configs):
        name, kw = parse(tag)
        seg = build(name, **kw)
        tiles, ms = [], []
        for i, r in enumerate(rows):
            im = cv2.imread(r["image"])
            t0 = time.perf_counter()
            res = seg(im)
            ms.append((time.perf_counter() - t0) * 1000)
            v = im.copy()
            th = max(2, im.shape[1] // 250)
            if res.ok:
                if res.iris_mask is not None:
                    cnts, _ = cv2.findContours(res.iris_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(v, cnts, -1, (255, 255, 0), th)
                for e, c in ((res.iris, (0, 255, 0)), (res.pupil, (255, 0, 255))):
                    cv2.ellipse(v, (int(e.cx), int(e.cy)), (int(e.a), int(e.b)), e.theta, 0, 360, c, th)
            v = cv2.resize(v, (240, 180))
            cv2.rectangle(v, (0, 0), (40, 18), (0, 0, 0), -1)
            cv2.putText(v, str(i), (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            tiles.append(v)
        grid = [np.hstack(tiles[j:j + 6]) for j in range(0, len(tiles), 6)]
        sheet = np.vstack(grid)
        cv2.imwrite(str(out / f"sheet_{ci:02d}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f"sheet_{ci:02d}.jpg  {tag}  median {np.median(ms):.0f} ms")
    with open(out / "index.csv", "w") as f:
        f.write("tile,id\n")
        for i, r in enumerate(rows):
            f.write(f"{i},{r['id']}\n")


if __name__ == "__main__":
    main()
