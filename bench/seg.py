"""Segmentation benchmark over a grid of segmenter configurations.

python -m bench.seg --config bench/configs/seg_grid.yaml --split test --out results/seg

Each worker process runs with 1 OpenCV/ORT thread so latencies are single-core numbers.
Per-image rows -> results/seg/per_image.csv; summary -> results/seg/summary.csv
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import yaml

from aeye_iris.geometry import ellipse_from_mask
from aeye_iris.segment import build
from bench.metrics import boundary_hd95, center_error, dice, nice1_error
from train.dataset import load_label, read_manifest

_SEG = {}


def expand(cfg: dict) -> list[dict]:
    """YAML entries: {name, params: {k: v or [list -> grid]}}."""
    out = []
    for e in cfg["segmenters"]:
        params = e.get("params", {})
        keys = list(params)
        vals = [v if isinstance(v, list) else [v] for v in params.values()]
        for combo in itertools.product(*vals):
            kw = dict(zip(keys, combo))
            tag = e["name"] + "".join(f"|{k}={v}" for k, v in kw.items())
            out.append({"tag": tag, "name": e["name"], "kw": kw})
    return out


def _init():
    cv2.setNumThreads(1)
    os.environ["OMP_NUM_THREADS"] = "1"


def _run(args):
    conf, row, root = args
    key = conf["tag"]
    if key not in _SEG:
        _SEG[key] = build(conf["name"], **conf["kw"])
    seg = _SEG[key]
    img = cv2.imread(str(Path(root) / row["image"]), cv2.IMREAD_COLOR)
    lab = load_label(row, Path(root))
    if lab.shape != img.shape[:2]:
        lab = cv2.resize(lab, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    seg(img)  # warm-up for the first call of each model is amortised below
    t0 = time.perf_counter()
    r = seg(img)
    ms = (time.perf_counter() - t0) * 1000
    gi, gp = lab == 1, lab == 2
    has_p = bool(row.get("pupil_mask"))
    out = {"tag": key, "id": row["id"], "dataset": row["dataset"], "ms": ms, "ok": int(r.ok)}
    gt_pupil_e = ellipse_from_mask(gp) if has_p and gp.sum() > 20 else None
    norm = np.sqrt(max(gi.sum() + gp.sum(), 1) / np.pi)
    if r.ok:
        pi = r.iris_mask > 0
        out.update(dice_iris=dice(pi, gi), e1=nice1_error(pi, gi), hd95_iris=boundary_hd95(pi, gi))
        if has_p:
            pp = r.pupil_mask > 0
            out.update(dice_pupil=dice(pp, gp), pupil_center_err=center_error(r.pupil, gt_pupil_e, norm))
        out["iris_r_rel_err"] = float(abs(r.iris.r - norm) / norm) if norm > 0 else np.nan
    else:
        out.update(dice_iris=0.0, e1=float(gi.mean()), hd95_iris=np.nan)
        if has_p:
            out.update(dice_pupil=0.0, pupil_center_err=np.nan)
    return out


def summarize(rows: list[dict]) -> list[dict]:
    by = {}
    for r in rows:
        by.setdefault((r["tag"], r["dataset"]), []).append(r)
    out = []
    for (tag, ds), rs in sorted(by.items()):
        def col(k):
            return np.array([r.get(k, np.nan) for r in rs], float)
        di = col("dice_iris")
        dp = col("dice_pupil")
        out.append({
            "tag": tag, "dataset": ds, "n": len(rs),
            "dice_iris_mean": np.nanmean(di), "dice_iris_median": np.nanmedian(di),
            "fail_rate_dice<0.8": float((di < 0.8).mean()),
            "e1_mean": np.nanmean(col("e1")), "hd95_median": np.nanmedian(col("hd95_iris")),
            "dice_pupil_mean": np.nanmean(dp) if np.isfinite(dp).any() else np.nan,
            "pupil_center_err_median": np.nanmedian(col("pupil_center_err")) if np.isfinite(col("pupil_center_err")).any() else np.nan,
            "ms_p50": np.percentile(col("ms"), 50), "ms_p95": np.percentile(col("ms"), 95),
            "seg_ok_rate": float(col("ok").mean()),
        })
    return out


def write_csv(path: Path, rows: list[dict]):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--root", default=".")
    ap.add_argument("--split", default="test")
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="results/seg")
    ap.add_argument("--only", default=None, help="substring filter on config tags")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    confs = expand(cfg)
    if a.only:
        confs = [c for c in confs if a.only in c["tag"]]
    rows = read_manifest(a.manifest, a.split, a.datasets)
    if a.limit:
        rng = np.random.default_rng(0)
        by = {}
        for r in rows:
            by.setdefault(r["dataset"], []).append(r)
        rows = []
        for ds, rs in by.items():
            idx = rng.choice(len(rs), min(a.limit, len(rs)), replace=False)
            rows += [rs[i] for i in sorted(idx)]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"{len(confs)} configs x {len(rows)} images", flush=True)
    results = []
    with ProcessPoolExecutor(a.workers, initializer=_init) as ex:
        for conf in confs:
            t0 = time.time()
            # chunk by config so each worker builds a model once
            res = list(ex.map(_run, [(conf, r, a.root) for r in rows], chunksize=8))
            results += res
            s = summarize(res)
            for x in s:
                print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in x.items()}), flush=True)
            print(f"  [{time.time()-t0:.0f}s]", flush=True)
    write_csv(out / "per_image.csv", results)
    write_csv(out / "summary.csv", summarize(results))


if __name__ == "__main__":
    main()
