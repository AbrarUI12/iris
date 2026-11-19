"""Marking-candidate benchmark (SYNTHETIC GROUND TRUTH).

A. Injection on real irises: for each held-out real image, build the normalised iris
   from the GROUND-TRUTH segmentation, inject one marking of a given type/strength with
   a known mask, run every (method, k) and match candidates (IoU >= 0.1).
   Also run each detector on the un-injected iris -> false positives per image on real
   texture (some of these may be genuine natural markings; reported as an upper bound).
B. Collarette radius error and fibre-orientation accuracy on procedurally rendered eyes
   (known collarette radius / fibre direction).

python -m bench.markings --out results/markings --limit 150
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from aeye_iris.geometry import ellipse_from_mask
from aeye_iris.markings.detectors import DETECTORS, collarette, fibre_orientation
from aeye_iris.normalize import normalize
from aeye_iris.segment.learned import _limbus_from_mask
from aeye_iris.types import SegResult
from bench.metrics import iou, prf
from bench.seg import write_csv
from synth.eye import render_eye
from synth.markings import inject
from train.dataset import load_label, read_manifest

KS = (2.0, 2.5, 3.0, 3.5, 4.0)
STRICT = (False, True)
STRENGTHS = (0.5, 1.0)
NORMS = ((64, 512),)  # 32x256 was worse for every marking in round 1 (results kept in git history)


def gt_seg(lab: np.ndarray) -> SegResult | None:
    pm = (lab == 2).astype(np.uint8)
    union = (lab > 0).astype(np.uint8)
    pupil = ellipse_from_mask(pm) if pm.sum() > 20 else None
    limbus = _limbus_from_mask(union, pupil.center if pupil else None)
    if pupil is None or limbus is None:
        return None
    return SegResult(pupil, limbus, (lab == 1).astype(np.uint8), pm, "gt")


def _one(args):
    row, root, seed, split = args
    cv2.setNumThreads(1)
    img = cv2.imread(str(Path(root) / row["image"]))
    if row.get("iris_mask"):
        seg = gt_seg(load_label(row, Path(root)))
    else:  # UPOL: no masks -> geometry from the classical segmenter (visually spot-checked)
        from aeye_iris.segment import build
        seg = build("daugman", channel="red", clahe=True, inpaint=True)(img)
        seg = seg if seg.ok else None
    out = []
    if seg is None:
        return out
    rng = np.random.default_rng(seed)
    for n_r, n_t in NORMS:
        norm = normalize(img, seg, n_r, n_t)
        if norm.valid.mean() < 0.4:
            continue
        base = norm.image.copy()
        for kind, (fn, methods) in DETECTORS.items():
            # clean image FP
            for m, k, st in itertools.product(methods, KS, STRICT):
                norm.image = base
                t0 = time.perf_counter()
                c = fn(norm, m, k=k, strict=st)
                ms = (time.perf_counter() - t0) * 1000
                out.append(dict(kind=kind, method=m, k=k, strict=st, strength=0.0, norm=f"{n_r}x{n_t}",
                                tp=0, fp=len(c), fn=0, ms=ms, id=row["id"], split=split))
            for s in STRENGTHS:
                inj, gmask, params = inject(base, norm.valid, kind, rng, s)
                if not gmask.any():
                    continue
                for m, k, st in itertools.product(methods, KS, STRICT):
                    norm.image = inj
                    c = fn(norm, m, k=k, strict=st)
                    # a detection overlapping the injected mask is a TP; others are FP
                    hit = [iou(cc.mask, gmask) for cc in c]
                    tp = int(any(h >= 0.1 for h in hit))
                    out.append(dict(kind=kind, method=m, k=k, strict=st, strength=s, norm=f"{n_r}x{n_t}",
                                    tp=tp, fp=len(c) - tp, fn=1 - tp,
                                    loc_iou=max(hit) if tp else np.nan, id=row["id"], split=split))
        norm.image = base
    return out


def summarize(rows):
    """Plain metrics. Precision counts EVERY non-matching detection on an injected image as a
    false positive (some may be genuine natural markings in the real host iris, so this is a
    lower bound on true precision)."""
    by = {}
    for r in rows:
        by.setdefault((r["split"], r["kind"], r["method"], r["k"], r["strict"]), []).append(r)
    out = []
    for (split, kind, m, k, st), rs in sorted(by.items(), key=lambda x: str(x[0])):
        clean = [r for r in rs if r["strength"] == 0]
        d = {"split": split, "kind": kind, "method": m, "k": k, "strict": st,
             "fp_per_clean_image": float(np.mean([r["fp"] for r in clean])) if clean else np.nan,
             "ms_mean": float(np.mean([r["ms"] for r in clean])) if clean else np.nan}
        for s in STRENGTHS:
            inj = [r for r in rs if r["strength"] == s]
            if not inj:
                continue
            tp = sum(r["tp"] for r in inj)
            fp = sum(r["fp"] for r in inj)
            fn = sum(r["fn"] for r in inj)
            p, rec, f = prf(tp, fp, fn)
            d[f"precision@s{s}"], d[f"recall@s{s}"], d[f"f1@s{s}"] = p, rec, f
            d[f"loc_iou@s{s}"] = float(np.nanmean([r.get("loc_iou", np.nan) for r in inj])) if tp else np.nan
        out.append(d)
    return out


def select_on_val(summary):
    """Per marking: (method, k, strict) with the best F1 at full strength on VAL hosts;
    returns the matching TEST rows (the reported numbers)."""
    chosen = []
    for kind in sorted({d["kind"] for d in summary}):
        val = [d for d in summary if d["split"] == "val" and d["kind"] == kind]
        best = max(val, key=lambda d: (d.get("f1@s1.0", 0), -d["fp_per_clean_image"]))
        test = next(d for d in summary if d["split"] == "test" and d["kind"] == kind and d["method"] == best["method"]
                    and d["k"] == best["k"] and d["strict"] == best["strict"])
        chosen.append({**test, "val_f1@s1.0": best.get("f1@s1.0")})
    return chosen


def synthetic_struct(n: int = 60):
    """Collarette radius MAE and fibre orientation accuracy on rendered eyes."""
    rows = []
    rng = np.random.default_rng(0)
    for i in range(n):
        cf = float(rng.uniform(0.25, 0.5))
        fibre = "radial" if i % 2 == 0 else "transverse"
        img, gt = render_eye(seed=1000 + i, color=["blue", "brown", "green_mixed"][i % 3],
                             collarette_frac=cf, fibre=fibre, glare=bool(i % 3))
        seg = SegResult(gt["pupil"], gt["iris"], gt["iris_mask"], gt["pupil_mask"], "gt")
        for n_r, n_t in NORMS:
            norm = normalize(img, seg, n_r, n_t)
            for m in ("gradient", "texture"):
                c = collarette(norm, m)
                rows.append(dict(task="collarette", method=m, norm=f"{n_r}x{n_t}",
                                 err=abs(c.get("radius_frac_median", np.nan) - cf) if c else np.nan))
            for m in ("structure_tensor", "gabor"):
                fo = fibre_orientation(norm, m)
                pred = "radial" if fo.get("radial_index", 0) > 0 else "transverse"
                rows.append(dict(task="fibre", method=m, norm=f"{n_r}x{n_t}", correct=float(pred == fibre),
                                 index=fo.get("radial_index", np.nan)))
    summ = []
    by = {}
    for r in rows:
        by.setdefault((r["task"], r["method"], r["norm"]), []).append(r)
    for (t, m, nm), rs in sorted(by.items()):
        if t == "collarette":
            e = np.array([r["err"] for r in rs])
            summ.append(dict(task=t, method=m, norm=nm, radius_frac_mae=float(np.nanmean(e)),
                             within_005=float(np.nanmean(e < 0.05)), n=len(rs)))
        else:
            summ.append(dict(task=t, method=m, norm=nm, accuracy=float(np.mean([r["correct"] for r in rs])), n=len(rs)))
    return summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--root", default=".")
    ap.add_argument("--split", default="test")
    ap.add_argument("--datasets", nargs="*", default=["upol"])
    ap.add_argument("--limit", type=int, default=150)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="results/markings")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    res = []
    for split in ("val", "test"):
        rows = read_manifest(a.manifest, split, a.datasets)
        rng = np.random.default_rng(0)
        rows = [rows[i] for i in sorted(rng.choice(len(rows), min(a.limit, len(rows)), replace=False))]
        with ProcessPoolExecutor(a.workers) as ex:
            for r in ex.map(_one, [(r, a.root, i, split) for i, r in enumerate(rows)], chunksize=4):
                res += r
    print(f"injection runs: {len(res)} in {time.time()-t0:.0f}s", flush=True)
    write_csv(out / "injection_raw.csv", res)
    s = summarize(res)
    write_csv(out / "injection_summary.csv", s)
    ss = synthetic_struct()
    write_csv(out / "structure_summary.csv", ss)
    for x in ss:
        print(json.dumps(x))
    chosen = select_on_val(s)
    write_csv(out / "selected_test.csv", chosen)
    for d in chosen:
        print(json.dumps({kk: (round(v, 3) if isinstance(v, float) else v) for kk, v in d.items()}))


if __name__ == "__main__":
    main()
