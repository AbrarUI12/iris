"""Shared machinery for segmenters: polar sampling, occlusion masking, registry."""
from __future__ import annotations

from typing import Callable, Protocol

import cv2
import numpy as np

from ..geometry import Ellipse, annulus_mask
from ..types import SegResult


class Segmenter(Protocol):
    name: str

    def __call__(self, bgr: np.ndarray) -> SegResult: ...


REGISTRY: dict[str, Callable[..., Segmenter]] = {}


def register(name: str):
    def deco(cls):
        REGISTRY[name] = cls
        cls.name = name
        return cls
    return deco


def build(name: str, **kw) -> Segmenter:
    if name not in REGISTRY:
        # lazy imports so optional deps (onnxruntime models, mediapipe) are not required
        from . import classical, learned  # noqa: F401
    return REGISTRY[name](**kw)


# ----------------------------------------------------------------------------- polar
def circular_profiles(img: np.ndarray, centers: np.ndarray, radii: np.ndarray,
                      angles: np.ndarray) -> np.ndarray:
    """Mean intensity on circles: returns (n_centers, n_radii).

    img float32 single channel; centers (N,2) xy; radii (R,); angles (A,) radians.
    """
    N, R, A = len(centers), len(radii), len(angles)
    ca, sa = np.cos(angles), np.sin(angles)
    mx = centers[:, 0, None, None] + radii[None, :, None] * ca[None, None, :]
    my = centers[:, 1, None, None] + radii[None, :, None] * sa[None, None, :]
    mx = mx.reshape(N * R, A).astype(np.float32)
    my = my.reshape(N * R, A).astype(np.float32)
    # remap handles max 32767 per dim; chunk rows
    out = np.empty((N * R, A), np.float32)
    step = 32000
    for i in range(0, N * R, step):
        out[i:i + step] = cv2.remap(img, mx[i:i + step], my[i:i + step], cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_REPLICATE)
    return out.reshape(N, R, A).mean(axis=2)


def ray_profiles(img: np.ndarray, center: tuple[float, float], radii: np.ndarray,
                 angles: np.ndarray) -> np.ndarray:
    """Intensity along rays: returns (A, R)."""
    cx, cy = center
    mx = (cx + np.outer(np.cos(angles), radii)).astype(np.float32)
    my = (cy + np.outer(np.sin(angles), radii)).astype(np.float32)
    return cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def smooth1d(x: np.ndarray, sigma: float, axis: int = -1) -> np.ndarray:
    if sigma <= 0:
        return x
    from scipy.ndimage import gaussian_filter1d
    return gaussian_filter1d(x, sigma, axis=axis, mode="nearest")


# ----------------------------------------------------------------------------- occlusion
def occlusion_mask(bgr: np.ndarray, pupil: Ellipse, iris: Ellipse, spec: np.ndarray | None,
                   k: float = 3.0) -> np.ndarray:
    """Visible-iris mask = annulus minus colour outliers (eyelid skin, lashes, sclera
    spill) and specular reflections. Iris colour reference is estimated robustly from
    the lateral band (3 and 9 o'clock), which is rarely occluded.

    k: outlier threshold in robust-sigma units (benchmarked).
    """
    h, w = bgr.shape[:2]
    ann = annulus_mask((h, w), pupil, iris).astype(bool)
    if ann.sum() < 30:
        return ann.astype(np.uint8)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    yy, xx = np.mgrid[0:h, 0:w]
    ang = np.arctan2(yy - iris.cy, xx - iris.cx)
    lateral = ann & (np.abs(np.sin(ang)) < 0.5)
    ref_px = lab[lateral] if lateral.sum() > 30 else lab[ann]
    med = np.median(ref_px, axis=0)
    mad = np.median(np.abs(ref_px - med), axis=0) * 1.4826 + np.array([4.0, 2.0, 2.0])
    z = np.abs(lab - med) / mad
    # lightness matters most (lids are brighter, lashes darker); chroma second
    dist = np.sqrt(z[..., 0] ** 2 + 0.5 * z[..., 1] ** 2 + 0.5 * z[..., 2] ** 2)
    bad = (dist > k) & ann
    if spec is not None:
        bad |= spec.astype(bool) & ann
    bad = cv2.morphologyEx(bad.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    # occlusion from lids is contiguous with the annulus top/bottom: close + fill
    bad = cv2.morphologyEx(bad, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    good = (ann & ~bad.astype(bool)).astype(np.uint8)
    # keep the largest component(s) touching the lateral band
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(good, connectivity=8)
    if n > 2:
        keep = np.zeros(n, bool)
        areas = stats[1:, cv2.CC_STAT_AREA]
        keep[1 + np.argmax(areas)] = True
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] > 0.05 * ann.sum():
                keep[i] = True
        good = keep[lbl].astype(np.uint8)
    return good


def finalize(bgr: np.ndarray, pupil: Ellipse | None, iris: Ellipse | None, spec, method: str,
             occl: bool = True, occl_k: float = 3.0, **extra) -> SegResult:
    h, w = bgr.shape[:2]
    if pupil is None or iris is None:
        return SegResult(None, None, None, None, method, dict(extra))
    if occl:
        im = occlusion_mask(bgr, pupil, iris, spec, occl_k)
    else:
        im = annulus_mask((h, w), pupil, iris)
    return SegResult(pupil, iris, im, pupil.to_mask((h, w)), method, dict(extra))
