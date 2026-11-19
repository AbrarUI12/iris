"""Capture-quality gate: explainable accept / retake / review decisions.

Two stages:
1. ``image_metrics`` – cheap whole-image checks run before segmentation (< 50 ms):
   focus (variance of Laplacian, Tenengrad), exposure, clipping, glare.
2. ``iris_metrics`` – after segmentation: iris size in pixels, visible fraction,
   iris-local focus and glare, geometric sanity.

Per-feature assessability: colour needs enough clean iris pixels; fine markings
(lacunae, furrows, fibres) additionally need resolution + local sharpness.
Thresholds live in ``QualityThresholds`` and are calibrated in bench/quality.py.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np

from .preprocess import specular_mask, resize_max
from .types import QualityOut, SegResult


@dataclass
class QualityThresholds:
    min_focus_lap: float = 30.0        # var(Laplacian) of iris region at 320px working size
    min_focus_global: float = 8.0      # var(Laplacian) of whole frame at 320px
    max_glare_iris: float = 0.12       # fraction of iris annulus covered by reflections
    max_clip_high: float = 0.08        # fraction of frame at >= 250
    max_clip_low: float = 0.35         # fraction of frame at <= 5
    min_mean_l: float = 25.0
    max_mean_l: float = 230.0
    min_iris_radius_px: float = 40.0   # colour OK below this? no -> retake
    min_iris_radius_fine: float = 90.0  # fine markings need a large iris in pixels
    min_visible_frac: float = 0.45     # visible iris / annulus
    min_visible_fine: float = 0.65
    min_focus_fine: float = 60.0
    pupil_ratio_range: tuple[float, float] = (0.1, 0.8)


def _lap_var(g: np.ndarray, mask: np.ndarray | None = None) -> float:
    lap = cv2.Laplacian(g, cv2.CV_32F, ksize=3)
    if mask is not None and mask.sum() > 50:
        return float(lap[mask > 0].var())
    return float(lap.var())


def _tenengrad(g: np.ndarray, mask: np.ndarray | None = None) -> float:
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    t = gx * gx + gy * gy
    if mask is not None and mask.sum() > 50:
        return float(np.sqrt(t[mask > 0].mean()))
    return float(np.sqrt(t.mean()))


def blockiness(gray: np.ndarray) -> float:
    """JPEG artefact score: mean |horizontal+vertical| step across 8x8 block boundaries
    divided by the mean step elsewhere (~1.0 for clean images, >1 for blocky ones).
    Must be computed at ORIGINAL resolution (resizing destroys the 8-px grid)."""
    g = gray.astype(np.float32)
    dx = np.abs(np.diff(g, axis=1))
    dy = np.abs(np.diff(g, axis=0))
    bx = dx[:, 7::8].mean() if dx.shape[1] > 8 else 0.0
    by = dy[7::8, :].mean() if dy.shape[0] > 8 else 0.0
    mx = np.ones(dx.shape[1], bool)
    mx[7::8] = False
    my = np.ones(dy.shape[0], bool)
    my[7::8] = False
    inner = (dx[:, mx].mean() + dy[my, :].mean()) / 2 + 1e-3
    return float(((bx + by) / 2) / inner)


def image_metrics(bgr: np.ndarray) -> dict[str, float]:
    small, _ = resize_max(bgr, 320)
    g = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    return {
        "blockiness": blockiness(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)),
        "focus_lap_global": _lap_var(g),
        "tenengrad_global": _tenengrad(g),
        "mean_l": float(g.mean()),
        "clip_high": float((g >= 250).mean()),
        "clip_low": float((g <= 5).mean()),
        "glare_global": float(specular_mask(small).mean()),
    }


def iris_metrics(bgr: np.ndarray, seg: SegResult) -> dict[str, float]:
    if not seg.ok:
        return {"seg_ok": 0.0}
    from .geometry import annulus_mask
    h, w = bgr.shape[:2]
    ann = annulus_mask((h, w), seg.pupil, seg.iris)
    vis = seg.iris_mask if seg.iris_mask is not None else ann
    # local sharpness at a resolution normalised to iris size (radius ~ 100 px)
    s = min(1.0, 100.0 / max(seg.iris.r, 1))
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if s < 1.0:
        g = cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        vm = cv2.resize(vis.astype(np.uint8), (g.shape[1], g.shape[0]), interpolation=cv2.INTER_NEAREST)
    else:
        vm = vis
    spec = specular_mask(bgr)
    vm_e = cv2.erode(vm.astype(np.uint8), np.ones((3, 3), np.uint8))
    return {
        "seg_ok": 1.0,
        "iris_radius_px": float(seg.iris.r),
        "pupil_ratio": float(seg.pupil.r / max(seg.iris.r, 1e-6)),
        "concentricity": float(np.hypot(seg.pupil.cx - seg.iris.cx, seg.pupil.cy - seg.iris.cy) / max(seg.iris.r, 1)),
        "visible_frac": float(vis.sum() / max(ann.sum(), 1)),
        "glare_iris": float((spec.astype(bool) & ann.astype(bool)).sum() / max(ann.sum(), 1)),
        "focus_lap_iris": _lap_var(g, vm_e),
        "tenengrad_iris": _tenengrad(g, vm_e),
        "iris_in_frame": float(seg.iris.cx - seg.iris.r >= -2 and seg.iris.cy - seg.iris.r >= -0.5 * seg.iris.r
                               and seg.iris.cx + seg.iris.r <= w + 2 and seg.iris.cy + seg.iris.r <= h + 0.5 * seg.iris.r),
    }


def precheck(bgr: np.ndarray, t: QualityThresholds = QualityThresholds()) -> QualityOut:
    """Fast pre-segmentation gate (the SRS '< 2 s' quality check)."""
    m = image_metrics(bgr)
    reasons = []
    if m["focus_lap_global"] < t.min_focus_global:
        reasons.append("blurred: hold the phone steady and tap to focus on the eye")
    if m["mean_l"] < t.min_mean_l or m["clip_low"] > t.max_clip_low:
        reasons.append("too dark: use more (indirect) light")
    if m["mean_l"] > t.max_mean_l or m["clip_high"] > t.max_clip_high:
        reasons.append("overexposed: avoid direct flash / bright light into the eye")
    status = "retake" if reasons else "accepted"
    return QualityOut(status=status, reasons=reasons, metrics={k: round(v, 4) for k, v in m.items()})


MODEL_FEATS = ["focus_lap_global", "tenengrad_global", "mean_l", "clip_high", "clip_low", "glare_global",
               "iris_radius_px", "visible_frac", "glare_iris", "focus_lap_iris", "tenengrad_iris", "seg_ok",
               "blockiness"]
LOG_FEATS = {"focus_lap_global", "tenengrad_global", "focus_lap_iris", "tenengrad_iris"}
MODEL_REASON = {  # feature -> user-facing retake message when the learned score fires
    "focus_lap_global": "image not sharp enough: hold steady and tap to focus on the eye",
    "focus_lap_iris": "iris detail not sharp enough: move slightly closer / refocus",
    "tenengrad_iris": "iris detail not sharp enough: move slightly closer / refocus",
    "tenengrad_global": "image not sharp enough: hold steady and tap to focus on the eye",
    "blockiness": "image is heavily compressed: send the original photo, not a screenshot/forward",
    "glare_iris": "reflections cover the iris: change the light angle",
    "glare_global": "strong reflections: avoid direct light into the eye",
    "mean_l": "lighting is too dark or too bright",
    "clip_high": "overexposed: avoid flash / direct light",
    "clip_low": "too dark: use more indirect light",
    "visible_frac": "iris partly covered: open the eye wider",
    "iris_radius_px": "iris too small: move the camera closer",
    "seg_ok": "iris not found: centre one open eye in the frame",
}


class QualityModel:
    """Logistic quality score (numpy inference) exported by bench/quality.py."""

    def __init__(self, d: dict):
        self.d = d
        self.mu, self.sd = np.array(d["mu"]), np.array(d["sd"])
        self.coef, self.b = np.array(d["coef"]), float(d["intercept"])
        self.thr = float(d["threshold"])

    @classmethod
    def load(cls, path):
        import json
        from pathlib import Path
        return cls(json.loads(Path(path).read_text()))

    @staticmethod
    def vector(m: dict) -> np.ndarray:
        v = np.array([float(m.get(f, 0.0)) if np.isfinite(m.get(f, 0.0)) else 0.0 for f in MODEL_FEATS])
        for i, f in enumerate(MODEL_FEATS):
            if f in LOG_FEATS:
                v[i] = np.log1p(max(v[i], 0.0))
        return v

    def score(self, m: dict) -> tuple[float, str]:
        z = (self.vector(m) - self.mu) / self.sd
        contrib = self.coef * z
        p = 1 / (1 + np.exp(-(contrib.sum() + self.b)))
        worst = MODEL_FEATS[int(np.argmax(contrib))]
        return float(p), worst


def assess(bgr: np.ndarray, seg: SegResult, t: QualityThresholds = QualityThresholds(),
           model: "QualityModel | None" = None) -> QualityOut:
    """Rule gate (explainable). With `model`, the learned score decides retake vs accept
    (better recall incl. JPEG, threshold set for ~10% false retakes on validation); the
    rule reasons are still reported, and a reason is derived from the top contributing
    feature when only the model fires."""
    q = _assess_rules(bgr, seg, t)
    if model is None or not seg.ok:
        return q
    p, worst = model.score(q.metrics)
    q.metrics["quality_risk"] = round(p, 4)
    if p >= model.thr:
        if q.status != "retake":
            q.reasons = [MODEL_REASON.get(worst, "photo quality too low: please retake")]
        q.status = "retake"
        q.color_assessable = q.fine_markings_assessable = False
    elif q.status == "retake":  # rules fired but the calibrated model says the photo is usable
        q.status = "review"
        q.reasons = [f"borderline quality ({r})" for r in q.reasons]
    return q


def _assess_rules(bgr: np.ndarray, seg: SegResult, t: QualityThresholds = QualityThresholds()) -> QualityOut:
    pre = precheck(bgr, t)
    m = dict(pre.metrics)
    reasons = list(pre.reasons)
    im = iris_metrics(bgr, seg)
    m.update({k: round(v, 4) for k, v in im.items()})
    if not seg.ok:
        reasons.append("iris not found: centre one open eye in the frame, looking at the camera")
        return QualityOut(status="retake", reasons=reasons, metrics=m,
                          color_assessable=False, fine_markings_assessable=False)
    if im["iris_radius_px"] < t.min_iris_radius_px:
        reasons.append("iris too small: move the camera closer")
    if im["visible_frac"] < t.min_visible_frac:
        reasons.append("iris partly covered: open the eye wider (eyelids/lashes cover the iris)")
    if im["glare_iris"] > t.max_glare_iris:
        reasons.append("reflections cover the iris: change the light angle")
    if im["focus_lap_iris"] < t.min_focus_lap:
        reasons.append("iris out of focus")
    lo, hi = t.pupil_ratio_range
    review = not (lo <= im["pupil_ratio"] <= hi) or im["concentricity"] > 0.3 or im["iris_in_frame"] < 1
    color_ok = not reasons
    fine_ok = (color_ok and im["iris_radius_px"] >= t.min_iris_radius_fine
               and im["visible_frac"] >= t.min_visible_fine and im["focus_lap_iris"] >= t.min_focus_fine)
    status = "retake" if reasons else ("review" if review else "accepted")
    if review and not reasons:
        reasons.append("unusual geometry: segmentation needs human review")
    return QualityOut(status=status, reasons=reasons, metrics=m,
                      color_assessable=color_ok, fine_markings_assessable=fine_ok)


def thresholds_dict(t: QualityThresholds) -> dict:
    return asdict(t)
