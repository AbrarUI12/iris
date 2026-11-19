"""Shared helpers for marking-candidate detectors on the normalised (polar) iris.

All detectors return ``Candidate`` objects in polar coordinates. A candidate is a
*visual* finding ("a region darker than its surroundings"), never an anatomical or
health classification; see docs in each module.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.ndimage import median_filter

from ..types import NormalizedIris


@dataclass
class Candidate:
    kind: str
    mask: np.ndarray  # polar bool mask (n_r x n_t)
    score: float
    attrs: dict = field(default_factory=dict)

    @property
    def bbox(self) -> list[int]:
        ys, xs = np.nonzero(self.mask)
        return [int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max())]

    @property
    def centroid(self) -> tuple[float, float]:
        ys, xs = np.nonzero(self.mask)
        # circular mean for theta
        n_t = self.mask.shape[1]
        ang = xs / n_t * 2 * np.pi
        c = np.arctan2(np.sin(ang).mean(), np.cos(ang).mean()) % (2 * np.pi)
        return float(ys.mean()), float(c / (2 * np.pi) * n_t)


def polar_lab(norm: NormalizedIris) -> np.ndarray:
    lab = cv2.cvtColor(norm.image, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[..., 0] *= 100 / 255
    lab[..., 1:] -= 128
    return lab


def wrap_pad(a: np.ndarray, p: int) -> np.ndarray:
    """Pad along theta (axis 1) circularly and along r (axis 0) by reflection."""
    a = np.concatenate([a[:, -p:], a, a[:, :p]], axis=1)
    return np.pad(a, ((p, p), (0, 0)) + ((0, 0),) * (a.ndim - 2), mode="reflect")


def unpad(a: np.ndarray, p: int) -> np.ndarray:
    return a[p:-p, p:-p]


def masked_background(x: np.ndarray, valid: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Robust local background (median) that ignores invalid pixels by filling them with
    the row median first. size = (rows, cols) window."""
    x = x.copy()
    for i in range(x.shape[0]):
        v = valid[i]
        x[i, ~v] = np.median(x[i, v]) if v.any() else np.median(x[valid]) if valid.any() else 0
    # median on a theta-downsampled grid (4x), then upsample: ~10x faster, same scale
    f = 4
    n_r, n_t = x.shape
    xs = x.reshape(n_r, n_t // f, f).mean(axis=2) if n_t % f == 0 else cv2.resize(x, (n_t // f, n_r), interpolation=cv2.INTER_AREA)
    sz = (size[0], max(3, size[1] // f | 1))
    p = max(sz)
    bg = unpad(median_filter(wrap_pad(xs, p), size=sz, mode="nearest"), p)
    return cv2.resize(bg.astype(np.float32), (n_t, n_r), interpolation=cv2.INTER_LINEAR)


def robust_sigma(x: np.ndarray) -> float:
    med = np.median(x)
    return float(np.median(np.abs(x - med)) * 1.4826 + 1e-6)


def components(mask: np.ndarray, valid: np.ndarray, min_area: int, max_area: int,
               kind: str, score_map: np.ndarray, extra=None) -> list[Candidate]:
    """Connected components -> candidates; merges components across the 0/360 seam."""
    m = (mask & valid).astype(np.uint8)
    n_t = m.shape[1]
    # seam handling: tile twice, label, keep components that start in the first copy
    tiled = np.concatenate([m, m], axis=1)
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(tiled, connectivity=8)
    out, seen = [], set()
    for i in range(1, n):
        x0 = stats[i, cv2.CC_STAT_LEFT]
        if x0 >= n_t:
            continue
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area or area > max_area:
            continue
        comp = lbl == i
        cm = comp[:, :n_t] | comp[:, n_t:]
        key = cm.tobytes()
        if key in seen:
            continue
        seen.add(key)
        sc = float(score_map[cm].mean())
        attrs = extra(cm) if extra else {}
        out.append(Candidate(kind, cm, sc, attrs))
    return out


def shape_attrs(cm: np.ndarray) -> dict:
    ys, xs = np.nonzero(cm)
    h = ys.max() - ys.min() + 1
    w = xs.max() - xs.min() + 1
    return {"area_px": int(cm.sum()), "radial_extent": int(h), "angular_extent": int(w),
            "elongation_radial": float(h / max(w, 1))}
