"""Iris colour benchmark.

Labels: data/labels/color_labels.csv (id,label) - hand-labelled by viewing iris crops
(blue / brown / green_mixed / uncertain); see data/DATASETS.md. Pixels come from the
GROUND-TRUTH visible-iris mask so this isolates the colour step from segmentation.

Compares feature sets x classifiers with subject-grouped 5-fold CV (macro-F1,
uncertain-rate, per-class confusion) and measures label-free stability: fraction of
images of the same eye (UBIRIS class) that receive that eye's majority prediction.

python -m bench.color --sheet          # write contact sheets for labelling
python -m bench.color                  # run benchmark
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from aeye_iris.color import (CLASSES, FEATURES, LearnedColor, classify_thresholds,
                             features, fit_learned, iris_pixels_lab)
from bench.markings import gt_seg
from bench.seg import write_csv
from train.dataset import load_label, read_manifest

LABELS = Path("data/labels/color_labels.csv")


def eye_key(row):
    return row["id"].rsplit("_", 1)[0]  # upol/001L_2 -> upol/001L (one eye)


_SEG = None


def load_lab(row, root, segmenter="daugman"):
    """Iris pixels in CIELAB. Uses GT masks when the dataset has them, otherwise the
    given segmenter (UPOL has no masks)."""
    global _SEG
    img = cv2.imread(str(Path(root) / row["image"]))
    if row.get("iris_mask"):
        seg = gt_seg(load_label(row, Path(root)))
    else:
        if _SEG is None:
            from aeye_iris.segment import build
            _SEG = build(segmenter, channel="red", clahe=True, inpaint=True)
        seg = _SEG(img)
        if not seg.ok:
            seg = None
    if seg is None:
        return None
    return iris_pixels_lab(img, seg.iris_mask, seg.pupil, seg.iris)


def sheet(rows, root, out: Path, n=240, per=40):
    out.mkdir(parents=True, exist_ok=True)
    seen, pick = set(), []
    for r in rows:
        k = eye_key(r)
        if k in seen or not r.get("pupil_mask"):
            continue
        seen.add(k)
        pick.append(r)
    rng = np.random.default_rng(0)
    pick = [pick[i] for i in sorted(rng.choice(len(pick), min(n, len(pick)), replace=False))]
    tiles = []
    for r in pick:
        img = cv2.imread(str(Path(root) / r["image"]))
        lab = load_label(r, Path(root))
        seg = gt_seg(lab)
        if seg is None:
            continue
        R = int(seg.iris.r * 1.15)
        x0, y0 = int(seg.iris.cx - R), int(seg.iris.cy - R)
        crop = cv2.copyMakeBorder(img, R, R, R, R, cv2.BORDER_CONSTANT)[y0 + R:y0 + 3 * R, x0 + R:x0 + 3 * R]
        crop = cv2.resize(crop, (140, 140))
        cv2.rectangle(crop, (0, 0), (139, 16), (0, 0, 0), -1)
        cv2.putText(crop, str(len(tiles)), (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        tiles.append((r["id"], crop))
    idx = []
    for s in range(0, len(tiles), per):
        chunk = tiles[s:s + per]
        rows_img = []
        for j in range(0, len(chunk), 8):
            row = [c for _, c in chunk[j:j + 8]]
            row += [np.zeros_like(row[0])] * (8 - len(row))
            rows_img.append(np.hstack(row))
        cv2.imwrite(str(out / f"sheet_{s // per:02d}.jpg"), np.vstack(rows_img), [cv2.IMWRITE_JPEG_QUALITY, 92])
    with open(out / "index.csv", "w") as f:
        f.write("tile,id\n")
        for i, (rid, _) in enumerate(tiles):
            f.write(f"{i},{rid}\n")
    print(f"{len(tiles)} tiles -> {out}")


def write_confusion(out: Path, y, pred, classes, title):
    """Saves confusion.csv, per_class.csv and confusion.png (rows = true, cols = predicted)."""
    cols = classes + (["uncertain"] if "uncertain" in pred else [])
    M = np.array([[sum(1 for a, b in zip(y, pred) if a == t and b == p) for p in cols] for t in classes])
    with open(out / "confusion.csv", "w") as f:
        f.write("true\\pred," + ",".join(cols) + "\n")
        for t, row in zip(classes, M):
            f.write(t + "," + ",".join(map(str, row)) + "\n")
    rows = []
    for c in classes:
        tp = sum(1 for a, b in zip(y, pred) if a == c and b == c)
        fp = sum(1 for a, b in zip(y, pred) if a != c and b == c)
        fn = sum(1 for a, b in zip(y, pred) if a == c and b != c)
        P = tp / (tp + fp) if tp + fp else 0.0
        R = tp / (tp + fn) if tp + fn else 0.0
        rows.append({"class": c, "support": tp + fn, "precision": P, "recall": R,
                     "f1": 2 * P * R / (P + R) if P + R else 0.0})
    rows.append({"class": "ACCURACY", "support": len(y),
                 "precision": float(np.mean([a == b for a, b in zip(y, pred)]))})
    write_csv(out / "per_class.csv", rows)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.6, 3.8), dpi=150)
    ax.imshow(M, cmap="Blues")
    ax.set_xticks(range(len(cols)), cols, rotation=30, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, int(M[i, j]), ha="center", va="center",
                    color="white" if M[i, j] > M.max() / 2 else "black", fontsize=10)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true (hand label)")
    ax.set_title(title, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "confusion.png")
    print("confusion (rows=true):", cols, M.tolist())


def macro_f1(y, p, classes):
    fs = []
    for c in classes:
        tp = sum((a == c) and (b == c) for a, b in zip(y, p))
        fp = sum((a != c) and (b == c) for a, b in zip(y, p))
        fn = sum((a == c) and (b != c) for a, b in zip(y, p))
        pr = tp / (tp + fp) if tp + fp else 0
        rc = tp / (tp + fn) if tp + fn else 0
        fs.append(2 * pr * rc / (pr + rc) if pr + rc else 0)
    return float(np.mean(fs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--root", default=".")
    ap.add_argument("--datasets", nargs="*", default=["upol"])
    ap.add_argument("--sheet", action="store_true")
    ap.add_argument("--out", default="results/color")
    a = ap.parse_args()
    rows_all = read_manifest(a.manifest, None, a.datasets)
    if a.sheet:
        sheet(rows_all, a.root, Path("data/labels/sheets"))
        return
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    byid = {r["id"]: r for r in rows_all}
    labels = {r["id"]: r["label"] for r in csv.DictReader(open(LABELS))}
    items = []
    for i, lbl in labels.items():
        if i in byid:
            lab = load_lab(byid[i], a.root)
            if lab is not None and len(lab) > 100:
                items.append((byid[i], lbl, lab))
    print(f"{len(items)} labelled iris images", {c: sum(l == c for _, l, _ in items) for c in (*CLASSES, 'uncertain')})
    sure = [(r, l, lab) for r, l, lab in items if l != "uncertain"]
    y = [l for _, l, _ in sure]
    groups = np.array([int(r["subject"]) if r["subject"].isdigit() else hash(r["subject"]) for r, _, _ in sure])
    folds = groups % 5
    res = []
    # thresholds (no training)
    for margin in (0.0, 0.15):
        p = [classify_thresholds(lab, margin)[0] for _, _, lab in sure]
        res.append(dict(feature="lab_rules", model="thresholds", margin=margin, macro_f1=macro_f1(y, p, CLASSES),
                        accuracy=float(np.mean([a == b for a, b in zip(y, p)])),
                        uncertain_rate=float(np.mean([q == "uncertain" for q in p])),
                        f1_on_decided=macro_f1([a for a, b in zip(y, p) if b != "uncertain"],
                                               [b for b in p if b != "uncertain"], CLASSES)))
    feats = {k: np.array([features(lab, k) for _, _, lab in sure]) for k in FEATURES}
    for kind in FEATURES:
        Xk = feats[kind]
        for model in ("logistic", "knn"):
            for margin in (0.0, 0.15):
                pred = [None] * len(y)
                for f in range(5):
                    tr, te = folds != f, folds == f
                    if te.sum() == 0:
                        continue
                    m = LearnedColor(fit_learned(Xk[tr], [y[i] for i in np.where(tr)[0]], kind, model))
                    for i in np.where(te)[0]:
                        pred[i] = m.predict(sure[i][2], margin)[0]
                res.append(dict(feature=kind, model=model, margin=margin, macro_f1=macro_f1(y, pred, CLASSES),
                                accuracy=float(np.mean([a == b for a, b in zip(y, pred)])),
                                uncertain_rate=float(np.mean([q == "uncertain" for q in pred])),
                                f1_on_decided=macro_f1([a for a, b in zip(y, pred) if b != "uncertain"],
                                                       [b for b in pred if b != "uncertain"], CLASSES)))
    # confusion matrix + per-class metrics of the SHIPPED setting (lab_median, logistic, margin 0)
    Xk = feats["lab_median"]
    cv_pred = [None] * len(y)
    for f in range(5):
        tr, te = folds != f, folds == f
        mdl = LearnedColor(fit_learned(Xk[tr], [y[i] for i in np.where(tr)[0]], "lab_median", "logistic"))
        for i in np.where(te)[0]:
            cv_pred[i] = mdl.predict(sure[i][2], 0.0)[0]
    write_confusion(out, y, cv_pred, list(CLASSES), "colour (lab_median logistic, subject-grouped 5-fold CV)")
    res.sort(key=lambda d: -d["macro_f1"])
    write_csv(out / "summary.csv", res)
    for d in res[:8]:
        print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in d.items()}))
    # fit best learned model on all labels and save; confusion matrix from CV
    best = next(d for d in res if d["model"] != "thresholds")
    params = fit_learned(feats[best["feature"]], y, best["feature"], best["model"])
    Path("models").mkdir(exist_ok=True)
    Path("models/color_model.json").write_text(json.dumps(params))
    # stability on unlabelled same-eye images (test split)
    model = LearnedColor(params)
    test = read_manifest(a.manifest, None, a.datasets)
    by_eye = {}
    for r in test:
        lab = load_lab(r, a.root)
        if lab is None or len(lab) < 100:
            continue
        by_eye.setdefault(eye_key(r), []).append((model.predict(lab, 0.0)[0], classify_thresholds(lab, 0.0)[0]))
    st = {}
    for j, name in ((0, "learned"), (1, "thresholds")):
        agree = [max(sum(p[j] == c for p in v) for c in CLASSES) / len(v) for v in by_eye.values() if len(v) >= 3]
        st[name] = float(np.mean(agree))
    (out / "stability.json").write_text(json.dumps({"eyes": len(by_eye), "majority_agreement": st,
                                                    "best": best}, indent=1))
    print("stability", st)


if __name__ == "__main__":
    main()
