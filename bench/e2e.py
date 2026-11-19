"""End-to-end pipeline benchmark: two-eye scans, latency (1 & 4 threads), memory, statuses.

python -m bench.e2e --pipelines bench/configs/e2e.yaml --out results/e2e
"""
from __future__ import annotations

import argparse
import copy
import json
import resource
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

from bench.seg import write_csv
from train.dataset import read_manifest


def run_one(cfg: dict, pairs, root, threads: int) -> dict:
    cv2.setNumThreads(threads)
    from aeye_iris.pipeline import Pipeline
    cfg = copy.deepcopy(cfg)
    if cfg["segmentation"]["name"] in ("onnx", "hybrid"):
        cfg["segmentation"].setdefault("params", {})["threads"] = threads
    pipe = Pipeline(cfg)
    times, pre_ms, statuses = [], [], {}
    cands = 0
    for a, b in pairs:
        L = cv2.imread(str(Path(root) / a["image"]))
        R = cv2.imread(str(Path(root) / b["image"]))
        t0 = time.perf_counter()
        obs = pipe.analyze({"left": L, "right": R})
        times.append((time.perf_counter() - t0) * 1000)
        for e in obs.eyes:
            statuses[e.quality.status] = statuses.get(e.quality.status, 0) + 1
            pre_ms.append(e.timings_ms.get("precheck", np.nan))
            cands += sum(m.status == "present_candidate" for m in e.markings)
    n_eyes = 2 * len(pairs)
    return {"threads": threads, "scan_ms_p50": float(np.percentile(times, 50)),
            "scan_ms_p95": float(np.percentile(times, 95)), "precheck_ms_p95": float(np.nanpercentile(pre_ms, 95)),
            "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            **{f"rate_{k}": v / n_eyes for k, v in statuses.items()},
            "candidates_per_eye": cands / n_eyes}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipelines", default="bench/configs/e2e.yaml")
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--root", default=".")
    ap.add_argument("--datasets", nargs="*", default=["ubiris_v2"])
    ap.add_argument("--pairs", type=int, default=40)
    ap.add_argument("--out", default="results/e2e")
    ap.add_argument("--_child", default=None)
    a = ap.parse_args()
    rows = read_manifest(a.manifest, "test", a.datasets)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(rows), 2 * a.pairs, replace=False)
    pairs = [(rows[idx[2 * i]], rows[idx[2 * i + 1]]) for i in range(a.pairs)]
    if a._child:  # isolated process per (pipeline, threads) for clean RSS numbers
        name, threads = a._child.split("@")
        cfg = yaml.safe_load(open(a.pipelines))["pipelines"][name]
        print(json.dumps({"pipeline": name, **run_one(cfg, pairs, a.root, int(threads))}))
        return
    names = list(yaml.safe_load(open(a.pipelines))["pipelines"])
    res = []
    for n in names:
        for th in (1, 4):
            o = subprocess.run([sys.executable, "-m", "bench.e2e", "--pipelines", a.pipelines, "--manifest", a.manifest,
                                "--root", a.root, "--pairs", str(a.pairs), "--_child", f"{n}@{th}",
                                "--datasets", *a.datasets], capture_output=True, text=True)
            line = [l for l in o.stdout.splitlines() if l.startswith("{")]
            if not line:
                print(o.stderr[-2000:])
                continue
            d = json.loads(line[-1])
            print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in d.items()}), flush=True)
            res.append(d)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "summary.csv", res)


if __name__ == "__main__":
    main()
