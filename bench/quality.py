"""Quality-gate benchmark with controlled degradations of held-out real images.

Positives = degraded copies (defocus, motion, dark, overexposed, glare, jpeg, lowres)
Negatives = original images. NOTE: some UBIRIS originals are themselves noisy (it is a
"noisy iris" database), so false-retake rates on originals are pessimistic.

Compares: (a) each single metric (ROC-AUC per degradation), (b) the rule-based gate
with default thresholds, (c) the rule gate with thresholds calibrated so <=10% of
originals are rejected, (d) a logistic model on all metrics (subject-grouped CV).

python -m bench.quality --segmenter daugman --out results/quality
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path

import cv2
import numpy as np

from aeye_iris.quality import QualityThresholds, assess, image_metrics, iris_metrics
from aeye_iris.segment import build
from bench.seg import write_csv
from synth.degrade import KINDS, degrade
from train.dataset import read_manifest

_S = {}
from aeye_iris.quality import MODEL_FEATS as FEATS, QualityModel  # noqa: E402


def _one(args):
    row, root, segname, segkw, seed = args
    cv2.setNumThreads(1)
    if segname not in _S:
        _S[segname] = build(segname, **segkw)
    seg = _S[segname]
    img = cv2.imread(str(Path(root) / row["image"]))
    rng = np.random.default_rng(seed)
    out = []
    variants = [("original", 0.0, img)] + [(k, s, degrade(img, k, rng, s)) for k in KINDS for s in (0.5, 1.0)]
    for kind, sev, im in variants:
        r = seg(im)
        m = image_metrics(im)
        m.update(iris_metrics(im, r))
        q = assess(im, r)
        out.append({"id": row["id"], "subject": row["subject"], "kind": kind, "severity": sev,
                    "label": int(kind != "original"), "gate_default": int(q.status == "retake"),
                    **{f: m.get(f, np.nan) for f in FEATS}})
    return out


def auc(y, s):
    from sklearn.metrics import roc_auc_score
    ok = np.isfinite(s)
    if len(set(y[ok])) < 2:
        return np.nan
    a = roc_auc_score(y[ok], s[ok])
    return max(a, 1 - a)  # direction-agnostic


def _thresholds_at(orig, q: float) -> QualityThresholds:
    col = lambda k: np.array([r[k] for r in orig], float)  # noqa: E731
    return replace(
        QualityThresholds(),
        min_focus_global=float(np.nanquantile(col("focus_lap_global"), q)),
        min_focus_lap=float(np.nanquantile(col("focus_lap_iris"), q)),
        max_glare_iris=float(np.nanquantile(col("glare_iris"), 1 - q)),
        min_visible_frac=float(np.nanquantile(col("visible_frac"), q)),
        min_iris_radius_px=float(np.nanquantile(col("iris_radius_px"), q)),
        min_mean_l=float(np.nanquantile(col("mean_l"), q / 2)),
        max_mean_l=float(np.nanquantile(col("mean_l"), 1 - q / 2)),
        max_clip_high=float(np.nanquantile(col("clip_high"), 1 - q / 2)),
        max_clip_low=float(np.nanquantile(col("clip_low"), 1 - q / 2)),
    )


def calibrate(rows, target_frr: float = 0.10) -> QualityThresholds:
    """One shared per-check quantile q, binary-searched so the JOINT false-retake rate
    on ORIGINAL validation images (segmentation succeeded) is ~target_frr."""
    orig = [r for r in rows if r["kind"] == "original" and r["seg_ok"] == 1]
    lo, hi = 0.0, target_frr
    for _ in range(25):
        q = (lo + hi) / 2
        frr = np.mean([rule_gate(r, _thresholds_at(orig, q)) for r in orig])
        lo, hi = (q, hi) if frr < target_frr else (lo, q)
    return _thresholds_at(orig, lo)


def rule_gate(r: dict, t: QualityThresholds) -> int:
    if r["seg_ok"] != 1:
        return 1
    bad = (r["focus_lap_global"] < t.min_focus_global or r["mean_l"] < t.min_mean_l or r["mean_l"] > t.max_mean_l
           or r["clip_high"] > t.max_clip_high or r["clip_low"] > t.max_clip_low
           or r["iris_radius_px"] < t.min_iris_radius_px or r["visible_frac"] < t.min_visible_frac
           or r["glare_iris"] > t.max_glare_iris or r["focus_lap_iris"] < t.min_focus_lap)
    return int(bad)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--root", default=".")
    ap.add_argument("--datasets", nargs="*", default=["upol", "synth_eyes"])
    ap.add_argument("--segmenter", default="daugman")
    ap.add_argument("--segkw", default="{}")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="results/quality")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    segkw = json.loads(a.segkw)
    res = {}
    for split in ("val", "test"):
        rows = read_manifest(a.manifest, split, a.datasets)
        rng = np.random.default_rng(1)
        rows = [rows[i] for i in sorted(rng.choice(len(rows), min(a.limit, len(rows)), replace=False))]
        with ProcessPoolExecutor(a.workers) as ex:
            res[split] = [x for r in ex.map(_one, [(r, a.root, a.segmenter, segkw, i) for i, r in enumerate(rows)],
                                           chunksize=4) for x in r]
        write_csv(out / f"metrics_{split}.csv", res[split])
    # thresholds calibrated on VAL, evaluated on TEST
    t_cal = calibrate(res["val"])
    (out / "calibrated_thresholds.json").write_text(json.dumps(asdict(t_cal), indent=1))
    test = res["test"]
    y = np.array([r["label"] for r in test])
    summary = []
    for kind in KINDS:
        sel = np.array([r["kind"] in ("original", kind) for r in test])
        yy = y[sel]
        d = {"degradation": kind}
        for f in FEATS:
            d[f"auc_{f}"] = auc(yy, np.array([r[f] for r in test], float)[sel])
        for name, fn in (("default", lambda r: r["gate_default"]), ("calibrated", lambda r: rule_gate(r, t_cal))):
            pred = np.array([fn(r) for r in test])[sel]
            d[f"{name}_recall_s1"] = float(pred[(yy == 1) & (np.array([r['severity'] for r in test])[sel] == 1.0)].mean())
            d[f"{name}_recall_s0.5"] = float(pred[(yy == 1) & (np.array([r['severity'] for r in test])[sel] == 0.5)].mean())
        summary.append(d)
    orig = [r for r in test if r["kind"] == "original"]
    summary.append({"degradation": "ORIGINAL (false retake rate)",
                    "default_recall_s1": float(np.mean([r["gate_default"] for r in orig])),
                    "calibrated_recall_s1": float(np.mean([rule_gate(r, t_cal) for r in orig]))})
    # logistic on all metrics, trained on VAL, threshold for <=10% false retakes on VAL originals,
    # exported as the production gate model; evaluated on TEST
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    X = lambda rows: np.array([QualityModel.vector(r) for r in rows])  # noqa: E731
    Xv, yv = X(res["val"]), np.array([r["label"] for r in res["val"]])
    mu, sd = Xv.mean(0), Xv.std(0) + 1e-6
    clf = LogisticRegression(max_iter=5000, C=1.0, class_weight="balanced").fit((Xv - mu) / sd, yv)
    pv = clf.predict_proba((Xv - mu) / sd)[:, 1]
    thr = float(np.quantile(pv[yv == 0], 0.90))
    model = {"features": FEATS, "mu": mu.tolist(), "sd": sd.tolist(), "coef": clf.coef_[0].tolist(),
             "intercept": float(clf.intercept_[0]), "threshold": thr,
             "note": "trained on synthetic degradations of UPOL+synthetic val images; re-fit on A-EYE photos"}
    Path("models").mkdir(exist_ok=True)
    Path("models/quality_model.json").write_text(json.dumps(model, indent=1))
    qm = QualityModel(model)
    pt = np.array([qm.score(r)[0] for r in test])
    sev = np.array([r["severity"] for r in test])
    kinds = np.array([r["kind"] for r in test])
    for d in summary:
        k = d["degradation"]
        if k in KINDS:
            d["model_recall_s1"] = float((pt[(kinds == k) & (sev == 1.0)] >= thr).mean())
            d["model_recall_s0.5"] = float((pt[(kinds == k) & (sev == 0.5)] >= thr).mean())
    summary[-1]["model_recall_s1"] = float((pt[kinds == "original"] >= thr).mean())

    def prf_row(name, pred):
        tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
        P = tp / max(tp + fp, 1); R = tp / max(tp + fn, 1)
        return {"degradation": f"OVERALL {name}", "precision": P, "recall": R,
                "f1": 2 * P * R / max(P + R, 1e-9), "false_retake_rate": fp / max(fp + tn, 1),
                "accuracy": (tp + tn) / len(y)}
    summary.append(prf_row("default rules", np.array([r["gate_default"] for r in test])))
    summary.append(prf_row("calibrated rules", np.array([rule_gate(r, t_cal) for r in test])))
    summary.append(prf_row("logistic model (shipped)", (pt >= thr).astype(int)))
    cm_rows = []
    for name, pred in (("default rules", np.array([r["gate_default"] for r in test])),
                       ("calibrated rules", np.array([rule_gate(r, t_cal) for r in test])),
                       ("logistic model (shipped)", (pt >= thr).astype(int))):
        cm_rows.append({"gate": name,
                        "TP_degraded_retaken": int(((pred == 1) & (y == 1)).sum()),
                        "FN_degraded_accepted": int(((pred == 0) & (y == 1)).sum()),
                        "FP_good_retaken": int(((pred == 1) & (y == 0)).sum()),
                        "TN_good_accepted": int(((pred == 0) & (y == 0)).sum())})
    write_csv(out / "confusion.csv", cm_rows)
    summary.append({"degradation": "ALL (AUC)", "auc_logistic": float(roc_auc_score(y, pt)),
                    "auc_calibrated_rules": float(roc_auc_score(y, [rule_gate(r, t_cal) for r in test]))})
    write_csv(out / "summary.csv", summary)
    for s in summary:
        print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items()
                          if not k.startswith("auc_") or k in ("auc_logistic", "auc_calibrated_rules",
                                                                "auc_focus_lap_iris", "auc_focus_lap_global")}))


if __name__ == "__main__":
    main()
