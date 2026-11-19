"""Ellipse geometry helpers shared by segmenters, normalization and zones.

Convention: an ellipse is (cx, cy, a, b, theta_deg) with a/b the semi-axes along the
rotated x/y axes, theta in degrees (OpenCV convention, clockwise in image coords).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import cv2
import numpy as np


@dataclass(frozen=True)
class Ellipse:
    cx: float
    cy: float
    a: float  # semi-axis along rotated x
    b: float  # semi-axis along rotated y
    theta: float = 0.0  # degrees

    @classmethod
    def circle(cls, cx: float, cy: float, r: float) -> "Ellipse":
        return cls(float(cx), float(cy), float(r), float(r), 0.0)

    @property
    def r(self) -> float:
        """Equivalent radius (geometric mean of semi-axes)."""
        return float(np.sqrt(max(self.a, 1e-6) * max(self.b, 1e-6)))

    @property
    def center(self) -> tuple[float, float]:
        return (self.cx, self.cy)

    def scaled(self, s: float) -> "Ellipse":
        return Ellipse(self.cx * s, self.cy * s, self.a * s, self.b * s, self.theta)

    def to_dict(self) -> dict:
        return {k: round(float(v), 3) for k, v in asdict(self).items()}

    def radius_at(self, phi: np.ndarray) -> np.ndarray:
        """Distance from center to boundary along image-space angle phi (radians)."""
        t = np.deg2rad(self.theta)
        p = phi - t
        a, b = max(self.a, 1e-6), max(self.b, 1e-6)
        return (a * b) / np.sqrt((b * np.cos(p)) ** 2 + (a * np.sin(p)) ** 2)

    def point_at(self, phi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        r = self.radius_at(phi)
        return self.cx + r * np.cos(phi), self.cy + r * np.sin(phi)

    def to_mask(self, shape: tuple[int, int]) -> np.ndarray:
        m = np.zeros(shape[:2], np.uint8)
        if self.a <= 0 or self.b <= 0:
            return m
        cv2.ellipse(
            m,
            (int(round(self.cx * 16)), int(round(self.cy * 16))),
            (int(round(self.a * 16)), int(round(self.b * 16))),
            self.theta, 0, 360, 1, -1, cv2.LINE_8, 4,
        )
        return m

    def contains(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        t = np.deg2rad(self.theta)
        dx, dy = x - self.cx, y - self.cy
        u = dx * np.cos(t) + dy * np.sin(t)
        v = -dx * np.sin(t) + dy * np.cos(t)
        return (u / max(self.a, 1e-6)) ** 2 + (v / max(self.b, 1e-6)) ** 2 <= 1.0


def fit_ellipse_points(xs: np.ndarray, ys: np.ndarray) -> Ellipse | None:
    """Least-squares ellipse fit (OpenCV); falls back to circle fit for few points."""
    xs = np.asarray(xs, np.float32)
    ys = np.asarray(ys, np.float32)
    if len(xs) < 3:
        return None
    if len(xs) < 6:
        return fit_circle_points(xs, ys)
    pts = np.stack([xs, ys], 1).reshape(-1, 1, 2)
    (cx, cy), (w, h), ang = cv2.fitEllipseDirect(pts)
    if not np.isfinite([cx, cy, w, h]).all() or w <= 0 or h <= 0:
        return fit_circle_points(xs, ys)
    return Ellipse(float(cx), float(cy), float(w / 2), float(h / 2), float(ang))


def fit_circle_points(xs: np.ndarray, ys: np.ndarray) -> Ellipse | None:
    """Algebraic (Kasa) circle fit."""
    xs = np.asarray(xs, np.float64)
    ys = np.asarray(ys, np.float64)
    if len(xs) < 3:
        return None
    A = np.stack([xs, ys, np.ones_like(xs)], 1)
    rhs = xs**2 + ys**2
    sol, *_ = np.linalg.lstsq(A, rhs, rcond=None)
    cx, cy = sol[0] / 2, sol[1] / 2
    r2 = sol[2] + cx**2 + cy**2
    if r2 <= 0:
        return None
    return Ellipse.circle(cx, cy, np.sqrt(r2))


def ransac_ellipse(
    xs: np.ndarray, ys: np.ndarray, iters: int = 60, tol: float = 2.0,
    max_ecc: float = 1.6, rng: np.random.Generator | None = None,
) -> Ellipse | None:
    """Robust ellipse fit: RANSAC on circles (5-pt ellipse is too unstable on noisy
    edge points), then a final ellipse refit on the inliers. ``max_ecc`` bounds a/b."""
    xs = np.asarray(xs, np.float64)
    ys = np.asarray(ys, np.float64)
    n = len(xs)
    if n < 6:
        return fit_circle_points(xs, ys)
    rng = rng or np.random.default_rng(0)
    best_inl, best_cnt = None, -1
    for _ in range(iters):
        idx = rng.choice(n, 3, replace=False)
        c = fit_circle_points(xs[idx], ys[idx])
        if c is None:
            continue
        d = np.abs(np.hypot(xs - c.cx, ys - c.cy) - c.a)
        inl = d < tol
        cnt = int(inl.sum())
        if cnt > best_cnt:
            best_cnt, best_inl = cnt, inl
    if best_inl is None or best_cnt < 6:
        return fit_circle_points(xs, ys)
    e = fit_ellipse_points(xs[best_inl], ys[best_inl])
    c = fit_circle_points(xs[best_inl], ys[best_inl])
    if e is None or max(e.a, e.b) / max(min(e.a, e.b), 1e-6) > max_ecc:
        return c
    return e


def ellipse_from_mask(mask: np.ndarray, fill_holes: bool = True) -> Ellipse | None:
    """Fit an ellipse to the largest connected component of a binary mask."""
    m = (mask > 0).astype(np.uint8)
    if m.sum() < 10:
        return None
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if len(c) < 6:
        return None
    hull = cv2.convexHull(c) if fill_holes else c
    if len(hull) < 6:
        hull = c
    (cx, cy), (w, h), ang = cv2.fitEllipseDirect(hull.astype(np.float32))
    if w <= 0 or h <= 0:
        return None
    return Ellipse(float(cx), float(cy), float(w / 2), float(h / 2), float(ang))


def annulus_mask(shape: tuple[int, int], pupil: Ellipse, iris: Ellipse) -> np.ndarray:
    return ((iris.to_mask(shape) > 0) & (pupil.to_mask(shape) == 0)).astype(np.uint8)
