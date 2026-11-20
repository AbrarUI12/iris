"""Does the segmentation actually look like an eye? Real-photo sanity checks.

Added after the first real phone photos (dark brown irises, photos through tinted
glasses, room reflected on the cornea) were wrongly ACCEPTED: the whole dark iris was
taken for the 'pupil' and the sclera/skin around it for the 'iris'.

All measures are in CIELAB L* (0-100), a*, b* on the ORIGINAL image.
"""
from __future__ import annotations

import cv2
import numpy as np

from .geometry import Ellipse
from .types import SegResult


def _lab(bgr):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[..., 0] *= 100 / 255
    lab[..., 1:] -= 128
    return lab


def _rho_phi(shape, pupil: Ellipse, iris: Ellipse):
    h, w = shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    phi = np.arctan2(yy - iris.cy, xx - iris.cx)
    d = np.hypot(xx - iris.cx, yy - iris.cy)
    ri = iris.radius_at(phi)
    dp = np.hypot(xx - pupil.cx, yy - pupil.cy)
    rp = pupil.radius_at(np.arctan2(yy - pupil.cy, xx - pupil.cx))
    return d / np.maximum(ri, 1), dp / np.maximum(rp, 1), phi


def measures(bgr: np.ndarray, seg: SegResult, work: int = 400) -> dict[str, float]:
    if not seg.ok:
        return {}
    h, w = bgr.shape[:2]
    s = min(1.0, work / max(h, w))
    small = cv2.resize(bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else bgr
    P, I = seg.pupil.scaled(s), seg.iris.scaled(s)
    lab = _lab(small)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    rel_i, rel_p, phi = _rho_phi(small.shape, P, I)
    lateral = np.abs(np.sin(phi)) < 0.5
    ring = (rel_i < 0.92) & (rel_p > 1.15)
    pupil_in = rel_p < 0.7
    # exclude black aperture / vignette (clinical iris cameras, macro lenses) from "outside"
    outside = (rel_i > 1.08) & (rel_i < 1.35) & lateral & (L > 15)
    colour = float(np.median(np.hypot(a, b)[ring | outside])) if (ring | outside).any() else 0.0
    m = {"is_colour": 0.0, "iris_center_in_frame": float(0 <= seg.iris.cx < w and 0 <= seg.iris.cy < h),
         "iris_r_over_maxside": float(seg.iris.r / max(h, w)),
         "pupil_axis_ratio": float(max(seg.pupil.a, seg.pupil.b) / max(min(seg.pupil.a, seg.pupil.b), 1e-3)),
         "iris_axis_ratio": float(max(seg.iris.a, seg.iris.b) / max(min(seg.iris.a, seg.iris.b), 1e-3))}
    m["is_colour"] = float(colour > 2.0)  # NIR / grayscale photos have ~zero chroma
    if ring.sum() < 50 or outside.sum() < 30:
        m.update(ring_px=float(ring.sum()), outside_px=float(outside.sum()))
        return m
    Lr, Lo = float(np.median(L[ring])), float(np.median(L[outside]))
    Lp = float(np.median(L[pupil_in])) if pupil_in.sum() > 10 else Lr
    mid = 0.5 * (Lr + Lo)
    chroma = np.hypot(a, b)
    sclera_like = (L > max(mid, 55)) & (chroma < 15)
    # sclera white point: bright, low-chroma pixels just outside the limbus (lateral)
    sc = outside & (L > 50)
    sa = float(np.median(a[sc])) if sc.sum() > 20 else np.nan
    sb = float(np.median(b[sc])) if sc.sum() > 20 else np.nan
    # scene reflection on the cornea: iris pixels much brighter than the iris median
    refl = ring & (L > Lr + 20)
    m.update(L_ring=Lr, L_outside=Lo, L_pupil=Lp,
             ring_contrast=Lo - Lr,                      # iris should be darker than sclera
             pupil_contrast=Lr - Lp,                     # pupil darker than iris
             sclera_in_ring=float(sclera_like[ring].mean()),
             sclera_a=sa, sclera_b=sb,
             reflection_frac=float(refl[ring].mean()))
    return m


# thresholds (calibrated on UPOL / synthetic / NIR validation images, see bench/plausibility.py)
T = {"ring_contrast_min": 5.0, "sclera_in_ring_max": 0.40, "pupil_contrast_min": 3.0,
     "pupil_axis_ratio_max": 1.6, "iris_axis_ratio_max": 1.35, "iris_r_over_maxside_max": 0.45,
     "cast_a_min": -10.0, "cast_a_max": 20.0, "cast_b_max": 50.0, "reflection_frac_max": 0.50}


def reasons(m: dict, t: dict = T) -> list[str]:
    """Retake reasons (user-facing). Empty list = plausible."""
    if not m:
        return ["iris not found: centre one open eye in the frame, looking at the camera"]
    out = []
    if m.get("iris_center_in_frame", 1) < 1 or m.get("iris_r_over_maxside", 0) > t["iris_r_over_maxside_max"]:
        out.append("iris not found reliably: centre the eye in the frame, iris clearly visible")
    if m.get("iris_axis_ratio", 1) > t["iris_axis_ratio_max"] or m.get("pupil_axis_ratio", 1) > t["pupil_axis_ratio_max"]:
        out.append("iris outline not round: look straight into the camera")
    if "ring_contrast" in m and m.get("is_colour", 1) > 0:
        if m["ring_contrast"] < t["ring_contrast_min"] or m["sclera_in_ring"] > t["sclera_in_ring_max"]:
            out.append("iris boundary not found: very dark iris or poor light - use bright, indirect side light, "
                       "no glasses, eye wide open")
        # NOTE: a colour-cast retake was tried and removed: the band outside the limbus mixes
        # sclera with pink lid skin / vessels, so it flagged good clinical photos (a* 21-27) and
        # did not catch the tinted-glasses photos (a* -4..-6). sclera_a / sclera_b stay reported.
        if m["reflection_frac"] > t["reflection_frac_max"]:
            out.append("reflection on the eye: turn away from windows / screens / lamps")
    return out
