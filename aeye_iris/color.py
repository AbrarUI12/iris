"""Iris colour appearance category: blue / brown / green_mixed / uncertain.

Features are computed on the ORIGINAL (not contrast-enhanced) image, restricted to the
visible-iris mask minus specular highlights and a thin band at both boundaries
(pupil-edge darkening and limbal ring).

Classifiers:
* thresholds – hand-set rules on CIELAB hue/chroma/lightness (no training)
* knn / logistic – fitted on reviewed labels (scikit-learn), stored as JSON params so
  inference stays numpy-only.
The 'uncertain' band is returned when the winning-class confidence is low.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .geometry import Ellipse
from .preprocess import specular_mask

CLASSES = ("blue", "brown", "green_mixed")
FEATURES = ("lab_median", "lab_pct", "hue_hist")


def iris_pixels_lab(bgr: np.ndarray, iris_mask: np.ndarray, pupil: Ellipse | None = None,
                    iris: Ellipse | None = None, trim: float = 0.12) -> np.ndarray:
    m = iris_mask.astype(bool) & ~specular_mask(bgr).astype(bool)
    if pupil is not None and iris is not None and trim > 0:
        h, w = m.shape
        yy, xx = np.mgrid[0:h, 0:w]
        phi = np.arctan2(yy - iris.cy, xx - iris.cx)
        d = np.hypot(xx - iris.cx, yy - iris.cy)
        rp, ri = pupil.r, iris.radius_at(phi)
        rho = (d - rp) / np.maximum(ri - rp, 1)
        m &= (rho > trim) & (rho < 1 - trim)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).reshape(-1, 3)[m.ravel()].astype(np.float32)
    # to CIELAB units: L 0..100, a/b signed
    lab[:, 0] *= 100 / 255
    lab[:, 1:] -= 128
    return lab


def features(lab: np.ndarray, kind: str = "lab_pct") -> np.ndarray:
    if len(lab) < 20:
        return np.full(_dim(kind), np.nan, np.float32)
    L, a, b = lab[:, 0], lab[:, 1], lab[:, 2]
    if kind == "lab_median":
        med = np.median(lab, 0)
        return np.array([*med, np.hypot(med[1], med[2]), np.degrees(np.arctan2(med[2], med[1]))], np.float32)
    if kind == "lab_pct":
        out = []
        for ch in (L, a, b):
            out += list(np.percentile(ch, [10, 50, 90]))
        hue = np.degrees(np.arctan2(np.median(b), np.median(a)))
        out += [np.hypot(np.median(a), np.median(b)), hue]
        return np.array(out, np.float32)
    if kind == "hue_hist":
        chroma = np.hypot(a, b)
        hue = np.degrees(np.arctan2(b, a)) % 360
        w = np.clip(chroma / 20, 0, 1)
        hist, _ = np.histogram(hue, bins=12, range=(0, 360), weights=w)
        hist = hist / max(hist.sum(), 1e-6)
        return np.array([*hist, np.median(L) / 100, np.median(chroma) / 40,
                         float((b < 0).mean())], np.float32)
    raise ValueError(kind)


def _dim(kind):
    return {"lab_median": 5, "lab_pct": 11, "hue_hist": 15}[kind]


def classify_thresholds(lab: np.ndarray, uncertain_margin: float = 0.15) -> tuple[str, float]:
    """Training-free rules (CIELAB). Brown: warm hue + low lightness or high chroma.
    Blue: b* <= ~2 (cool/neutral) with low chroma. Green/mixed: yellow-green hue band or
    mixture of a warm inner band with a cool outer band."""
    if len(lab) < 20:
        return "uncertain", 0.0
    L, a, b = np.median(lab, 0)
    chroma = float(np.hypot(a, b))
    hue = float(np.degrees(np.arctan2(b, a)))
    frac_cool = float((lab[:, 2] < 2).mean())
    scores = {
        "blue": np.clip((frac_cool - 0.3) / 0.4, 0, 1) * np.clip((L - 25) / 20, 0, 1) + np.clip((2 - b) / 6, 0, 1),
        "brown": np.clip((65 - hue) / 20, 0, 1) * np.clip((chroma - 6) / 10, 0, 1) + np.clip((40 - L) / 15, 0, 1) * np.clip((b - 4) / 6, 0, 1),
        "green_mixed": np.clip((hue - 65) / 15, 0, 1) * np.clip((b - 2) / 6, 0, 1) + np.clip(1 - abs(frac_cool - 0.45) / 0.25, 0, 1) * 0.5,
    }
    s = np.array([scores[c] for c in CLASSES], np.float32)
    p = np.exp(3 * s) / np.exp(3 * s).sum()
    i = int(np.argmax(p))
    conf = float(p[i])
    if conf - np.sort(p)[-2] < uncertain_margin:
        return "uncertain", conf
    return CLASSES[i], conf


class LearnedColor:
    """kNN or logistic regression stored as plain arrays (numpy-only inference)."""

    def __init__(self, params: dict):
        self.p = params
        self.kind = params["feature"]
        self.mu = np.array(params["mu"], np.float32)
        self.sd = np.array(params["sd"], np.float32)

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text()))

    def predict(self, lab: np.ndarray, uncertain_margin: float = 0.15) -> tuple[str, float]:
        f = features(lab, self.kind)
        if not np.isfinite(f).all():
            return "uncertain", 0.0
        z = (f - self.mu) / self.sd
        if self.p["model"] == "logistic":
            W = np.array(self.p["coef"], np.float32)
            b = np.array(self.p["intercept"], np.float32)
            s = W @ z + b
            pr = np.exp(s - s.max())
            pr /= pr.sum()
        else:  # knn
            X = np.array(self.p["X"], np.float32)
            y = np.array(self.p["y"])
            d = np.linalg.norm(X - z, axis=1)
            nn = np.argsort(d)[: self.p["k"]]
            pr = np.array([(y[nn] == c).mean() for c in self.p["classes"]], np.float32)
        classes = self.p["classes"]
        i = int(np.argmax(pr))
        srt = np.sort(pr)
        if len(srt) > 1 and srt[-1] - srt[-2] < uncertain_margin:
            return "uncertain", float(pr[i])
        return classes[i], float(pr[i])


def fit_learned(X: np.ndarray, y: list[str], kind: str, model: str = "logistic", k: int = 5) -> dict:
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = (X - mu) / sd
    classes = sorted(set(y))
    out = {"feature": kind, "model": model, "mu": mu.tolist(), "sd": sd.tolist(), "classes": classes}
    if model == "logistic":
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(C=1.0, max_iter=2000).fit(Z, y)
        coef = clf.coef_
        inter = clf.intercept_
        if len(classes) == 2:  # binary -> 2-row form
            coef = np.vstack([-coef[0], coef[0]]) / 2
            inter = np.array([-inter[0], inter[0]]) / 2
        out.update(coef=np.asarray(coef).tolist(), intercept=np.asarray(inter).tolist())
    else:
        out.update(X=Z.tolist(), y=list(y), k=k)
    return out
