"""Classical (training-free) iris/pupil localisation baselines.

* daugman  – integro-differential operator, coarse-to-fine grid search.
* hough    – Canny + Hough circle candidates re-ranked by boundary contrast.
* raycast  – dark-seed + radial edge ray casting + RANSAC ellipse (handles off-axis).

All work on a downscaled single-channel image (see preprocess.prepare) and return
ellipses in original pixel coordinates. The limbus is searched first (strongest edge in
visible light), then the pupil inside it.
"""
from __future__ import annotations

import cv2
import numpy as np

from ..geometry import Ellipse, ransac_ellipse
from ..preprocess import prepare, resize_max
from ..types import SegResult
from .base import circular_profiles, finalize, ray_profiles, register, smooth1d

DEG = np.pi / 180.0
# lateral arcs used for the limbus (avoid eyelids at top and bottom)
LIMBUS_ANGLES = np.concatenate([np.arange(-30, 46, 5), np.arange(135, 211, 5)]) * DEG
# pupil: everything except the top sector (upper lashes)
PUPIL_ANGLES = np.concatenate([np.arange(-50, 231, 10)]) * DEG


class _Base:
    def __init__(self, channel: str = "red", clahe: bool = False, inpaint: bool = True,
                 work_size: int = 240, rmin_frac: float = 0.10, rmax_frac: float = 0.48,
                 occl: bool = True, occl_k: float = 3.0, pupil_min: float = 0.12,
                 pupil_max: float = 0.75):
        self.channel, self.clahe, self.inpaint = channel, clahe, inpaint
        self.work_size = work_size
        self.rmin_frac, self.rmax_frac = rmin_frac, rmax_frac
        self.occl, self.occl_k = occl, occl_k
        self.pupil_min, self.pupil_max = pupil_min, pupil_max

    def tag(self) -> str:
        return (f"{self.name}[{self.channel}{'+clahe' if self.clahe else ''}"
                f"{'+inp' if self.inpaint else ''}{'+occ' if self.occl else ''}]")

    def __call__(self, bgr: np.ndarray) -> SegResult:
        g_full, clean, spec = prepare(bgr, self.channel, self.clahe, self.inpaint)
        g, s = resize_max(g_full, self.work_size)
        g = cv2.GaussianBlur(g, (0, 0), 1.0).astype(np.float32)
        try:
            limbus, pupil = self.locate(g)
        except Exception as e:  # pragma: no cover - defensive
            return SegResult(None, None, method=self.tag(), extra={"error": repr(e)})
        if limbus is None or pupil is None:
            return SegResult(None, None, method=self.tag())
        return finalize(clean, pupil.scaled(1 / s), limbus.scaled(1 / s), spec, self.tag(),
                        self.occl, self.occl_k)

    def r_range(self, g: np.ndarray) -> tuple[float, float]:
        m = min(g.shape[:2])
        return self.rmin_frac * m, self.rmax_frac * m

    def locate(self, g: np.ndarray) -> tuple[Ellipse | None, Ellipse | None]:
        raise NotImplementedError


# ----------------------------------------------------------------------------- Daugman
SPAN_PX = 4.0  # edge-score span at work resolution: soft limbus vs sharp pupil edge


def _span_diff(prof: np.ndarray, radii: np.ndarray, sigma: float) -> tuple[np.ndarray, int]:
    """Integro-differential edge score: mean(profile just outside) - mean(just inside)."""
    step = float(radii[1] - radii[0]) if len(radii) > 1 else 1.0
    k = max(1, int(round(SPAN_PX / step)))
    p = smooth1d(prof, sigma, axis=1)
    if p.shape[1] <= 2 * k:
        return np.diff(p, axis=1), 0
    return p[:, 2 * k:] - p[:, :-2 * k], k


def _ido_search(g, centers, radii, angles, sigma=1.5, sign=+1):
    """Grid search for the circle with the strongest outward edge. The edge score is the
    MIN over the left and right halves of the arc set, so a circle must sit on a real
    boundary on both sides (prevents half-on-pupil / half-on-lid solutions)."""
    left = np.cos(angles) < 0
    groups = [angles[left], angles[~left]] if left.any() and (~left).any() else [angles]
    d, k = None, 0
    for grp in groups:
        prof = circular_profiles(g, centers, radii, grp)
        dg, k = _span_diff(prof, radii, sigma)
        dg = dg * sign
        d = dg if d is None else np.minimum(d, dg)
    ci, ri = np.unravel_index(np.argmax(d), d.shape)
    r = radii[ri + k] if k else (radii[ri] + radii[ri + 1]) / 2
    return centers[ci], float(r), float(d[ci, ri])


def _grid(x0, x1, y0, y1, step):
    xs = np.arange(x0, x1 + 1e-6, step)
    ys = np.arange(y0, y1 + 1e-6, step)
    X, Y = np.meshgrid(xs, ys)
    return np.stack([X.ravel(), Y.ravel()], 1).astype(np.float32)


def _resolve_pair(g, cx, cy, r, rmax, pmin, pmax):
    """Disambiguate a strong circle C=(cx,cy,r): is it the limbus (pupil inside) or the
    pupil (limbus outside)? Both companions are searched and the stronger edge wins.
    Returns (limbus_xyR, pupil_xyR_or_None, c_was_pupil)."""
    # A: C is the limbus -> pupil inside
    off = 0.2 * r
    cA = _grid(cx - off, cx + off, cy - off, cy + off, max(1.0, r / 25))
    rA = np.arange(max(2.0, pmin * r), pmax * r, 1.0, dtype=np.float32)
    (px, py), pr, scA = _ido_search(g, cA, rA, PUPIL_ANGLES, 1.0) if len(rA) > 3 else ((cx, cy), 0, -1e9)
    # B: C is the pupil -> limbus outside (R in [r/pmax, r/pmin])
    lo, hi = r / pmax, min(rmax, r / pmin)
    scB = -1e9
    if hi - lo > 4:
        offB = 0.2 * r
        cB = _grid(cx - offB, cx + offB, cy - offB, cy + offB, max(1.0, r / 10))
        rB = np.arange(lo, hi, 1.0, dtype=np.float32)
        (ox, oy), R, scB = _ido_search(g, cB, rB, LIMBUS_ANGLES, 1.5)
    if scB > scA:
        return (float(ox), float(oy), float(R)), (cx, cy, r), True
    return (cx, cy, r), (float(px), float(py), float(pr)), False


@register("daugman")
class Daugman(_Base):
    def locate(self, g):
        h, w = g.shape
        rmin, rmax = self.r_range(g)
        step = max(2.0, min(h, w) / 30)
        c = _grid(0.2 * w, 0.8 * w, 0.2 * h, 0.8 * h, step)
        radii = np.arange(rmin, rmax, 2.0, dtype=np.float32)
        (cx, cy), r, _ = _ido_search(g, c, radii, LIMBUS_ANGLES, 2.0)
        c = _grid(cx - step, cx + step, cy - step, cy + step, 1.0)
        radii = np.arange(max(rmin, r - 6), min(rmax, r + 6), 0.5, dtype=np.float32)
        (cx, cy), R, _ = _ido_search(g, c, radii, LIMBUS_ANGLES, 1.5)
        (cx, cy, R), (px, py, pr), _ = _resolve_pair(g, cx, cy, R, rmax, self.pupil_min, self.pupil_max)
        limbus = Ellipse.circle(cx, cy, R)
        c = _grid(px - 2, px + 2, py - 2, py + 2, 0.5)
        radii = np.arange(max(2, pr - 3), pr + 3, 0.5, dtype=np.float32)
        (px, py), pr, _ = _ido_search(g, c, radii, PUPIL_ANGLES, 1.0)
        return limbus, Ellipse.circle(px, py, pr)


# ----------------------------------------------------------------------------- Hough
def _boundary_score(g, cx, cy, r, angles, sign=+1, band=0.08):
    radii = np.linspace(r * (1 - band), r * (1 + band), 9, dtype=np.float32)
    p = circular_profiles(g, np.array([[cx, cy]], np.float32), radii, angles)[0]
    return float((p[-3:].mean() - p[:3].mean()) * sign)


@register("hough")
class Hough(_Base):
    def locate(self, g):
        h, w = g.shape
        rmin, rmax = self.r_range(g)
        g8 = np.clip(g, 0, 255).astype(np.uint8)
        cands = cv2.HoughCircles(g8, cv2.HOUGH_GRADIENT, dp=1, minDist=4, param1=60, param2=12,
                                 minRadius=int(rmin), maxRadius=int(rmax))
        if cands is None:
            return None, None
        best, bs = None, -1e9
        for x, y, r in cands[0][:40]:
            if not (0.1 * w < x < 0.9 * w and 0.1 * h < y < 0.9 * h):
                continue
            sc = _boundary_score(g, x, y, r, LIMBUS_ANGLES, +1)
            if sc > bs:
                best, bs = (x, y, r), sc
        if best is None:
            return None, None
        cx, cy, R = map(float, best)
        (cx, cy, R), _, _ = _resolve_pair(g, cx, cy, R, rmax, self.pupil_min, self.pupil_max)
        limbus = Ellipse.circle(cx, cy, R)
        x0, y0 = int(max(0, cx - R)), int(max(0, cy - R))
        crop = g8[y0:int(cy + R) + 1, x0:int(cx + R) + 1]
        pc = cv2.HoughCircles(crop, cv2.HOUGH_GRADIENT, dp=1, minDist=2, param1=40, param2=8,
                              minRadius=int(self.pupil_min * R), maxRadius=int(self.pupil_max * R))
        pupil, ps = None, -1e9
        if pc is not None:
            for x, y, r in pc[0][:40]:
                x, y = x + x0, y + y0
                if np.hypot(x - cx, y - cy) > 0.25 * R:
                    continue
                sc = _boundary_score(g, x, y, r, PUPIL_ANGLES, +1)
                if sc > ps:
                    pupil, ps = Ellipse.circle(x, y, r), sc
        if pupil is None:  # fallback: concentric, typical ratio
            pupil = Ellipse.circle(cx, cy, 0.35 * R)
        return limbus, pupil


# ----------------------------------------------------------------------------- ray casting
@register("raycast")
class RayCast(_Base):
    def __init__(self, n_rays: int = 72, iters: int = 2, **kw):
        super().__init__(**kw)
        self.n_rays, self.iters = n_rays, iters

    def _seed(self, g):
        h, w = g.shape
        k = int(min(h, w) / 6) | 1
        b = cv2.blur(g, (k, k))
        y0, y1, x0, x1 = int(0.2 * h), int(0.8 * h), int(0.2 * w), int(0.8 * w)
        sub = b[y0:y1, x0:x1]
        iy, ix = np.unravel_index(np.argmin(sub), sub.shape)
        return float(ix + x0), float(iy + y0)

    def _edges(self, g, c, rlo, rhi, angles, sign=+1, sigma=1.5):
        radii = np.arange(rlo, rhi, 0.5, dtype=np.float32)
        P = smooth1d(ray_profiles(g, c, radii, angles), sigma, axis=1)
        D = np.diff(P, axis=1) * sign
        j = np.argmax(D, axis=1)
        strength = D[np.arange(len(angles)), j]
        r = radii[j] + 0.25
        keep = strength > np.percentile(strength, 25)
        xs = c[0] + r * np.cos(angles)
        ys = c[1] + r * np.sin(angles)
        return xs[keep], ys[keep]

    def locate(self, g):
        rmin, rmax = self.r_range(g)
        c = self._seed(g)
        ang = np.linspace(-35, 50, self.n_rays // 2) * DEG
        ang = np.concatenate([ang, np.pi - ang])
        # 1) dominant circle from the strongest per-ray edges (pupil OR limbus)
        e = None
        for _ in range(self.iters):
            xs, ys = self._edges(g, c, rmin * 0.5, rmax, ang, +1, 2.0)
            e = ransac_ellipse(xs, ys, tol=max(1.5, 0.03 * rmax), max_ecc=1.35)
            if e is None:
                return None, None
            c = (e.cx, e.cy)
        # 2) decide what it is, 3) refit both boundaries with band-limited rays
        (lx, ly, lR), pup, _ = _resolve_pair(g, e.cx, e.cy, e.r, rmax, self.pupil_min, self.pupil_max)
        xs, ys = self._edges(g, (lx, ly), 0.85 * lR, min(rmax * 1.1, 1.15 * lR), ang, +1, 2.0)
        limbus = ransac_ellipse(xs, ys, tol=max(1.5, 0.03 * lR), max_ecc=1.3) or Ellipse.circle(lx, ly, lR)
        if abs(limbus.r - lR) > 0.15 * lR:
            limbus = Ellipse.circle(lx, ly, lR)
        px, py, pr = pup
        pang = np.linspace(-50, 230, self.n_rays) * DEG
        xs, ys = self._edges(g, (px, py), 0.8 * pr, 1.2 * pr, pang, +1, 1.0)
        pupil = ransac_ellipse(xs, ys, tol=max(1.0, 0.04 * pr), max_ecc=1.3)
        if pupil is None or abs(pupil.r - pr) > 0.2 * pr:
            pupil = Ellipse.circle(px, py, pr)
        return limbus, pupil
