"""Inject parametric markings into REAL normalised (polar) irises with exact GT masks.

Public datasets carry no labels for the SRS marking categories, so marking detectors
are benchmarked by injection: the host texture is a real UBIRIS iris (so false
positives come from real texture, glare residue and lids), the injected marking has a
known mask. Results are SYNTHETIC evidence: they show whether a detector can find a
marking of a given contrast/size, not that it matches practitioner judgement.
"""
from __future__ import annotations

import cv2
import numpy as np


def _soft_ellipse(shape, r0, c0, rr, rc, angle=0.0, wrap=True):
    n_r, n_t = shape
    yy, xx = np.mgrid[0:n_r, 0:n_t].astype(np.float32)
    dx = xx - c0
    if wrap:
        dx = (dx + n_t / 2) % n_t - n_t / 2
    dy = yy - r0
    t = np.deg2rad(angle)
    u = dx * np.cos(t) + dy * np.sin(t)
    v = -dx * np.sin(t) + dy * np.cos(t)
    d = (u / max(rc, 0.5)) ** 2 + (v / max(rr, 0.5)) ** 2
    alpha = np.clip(1.5 - d, 0, 1)  # soft edge
    return alpha, d <= 1.0


def _apply(img, alpha, fn):
    f = img.astype(np.float32)
    out = f * (1 - alpha[..., None]) + fn(f) * alpha[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


def inject(polar_bgr: np.ndarray, valid: np.ndarray, kind: str, rng: np.random.Generator,
           strength: float = 1.0) -> tuple[np.ndarray, np.ndarray, dict]:
    """Returns (new image, GT mask, params). Placement is restricted to valid pixels."""
    n_r, n_t = valid.shape
    for _ in range(50):
        if kind == "dark_pigmentation":
            rr, rc = rng.uniform(2, 5), rng.uniform(3, 10)
            r0 = rng.uniform(0.15, 0.85) * n_r
        elif kind == "white_marking":
            rr, rc = rng.uniform(2, 6), rng.uniform(2, 6)
            r0 = rng.uniform(0.15, 0.85) * n_r
        elif kind == "lacuna":
            rr, rc = rng.uniform(5, 10), rng.uniform(2.5, 5)  # radially elongated
            r0 = rng.uniform(0.3, 0.75) * n_r
        elif kind == "contraction_furrow":
            rr, rc = rng.uniform(0.8, 1.5), rng.uniform(25, 70)  # long thin arc
            r0 = rng.uniform(0.55, 0.9) * n_r
        else:
            raise ValueError(kind)
        c0 = rng.uniform(0, n_t)
        alpha, gt = _soft_ellipse((n_r, n_t), r0, c0, rr, rc)
        if gt.sum() and valid[gt].mean() > 0.9:
            break
    else:
        return polar_bgr, np.zeros_like(valid), {}
    s = strength
    if kind == "dark_pigmentation":
        # darker + warmer (brown/orange pigment): scale BGR unevenly
        fac = np.array([0.45, 0.6, 0.8], np.float32)
        fac = 1 - (1 - fac) * s
        out = _apply(polar_bgr, alpha, lambda f: f * fac)
    elif kind == "white_marking":
        out = _apply(polar_bgr, alpha * min(1.0, 0.75 * s),
                     lambda f: f + (235 - f) * 0.6)
    elif kind == "lacuna":
        out = _apply(polar_bgr, alpha, lambda f: f * (1 - 0.4 * s))
    else:  # furrow
        out = _apply(polar_bgr, alpha, lambda f: f * (1 - 0.35 * s))
    gt &= valid
    return out, gt, {"r0": float(r0), "c0": float(c0), "rr": float(rr), "rc": float(rc), "strength": s}
