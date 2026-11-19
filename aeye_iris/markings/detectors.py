"""Candidate detectors for the six SRS marking categories (visual definitions only).

Operational visual definitions used here (must be confirmed by practitioners):
* dark_pigmentation – compact region darker AND more chromatic/brown than local iris.
* white_marking     – compact/streak region brighter and less saturated than local iris
                      (specular reflections are excluded via the validity mask).
* lacuna            – closed, radially elongated, darker gap with little hue change
                      (structural, not pigment).
* contraction_furrow – thin dark concentric arc (constant radius, extended in angle),
                      usually in the outer half.
* collarette        – the jagged ring separating pupillary and ciliary zones; reported as
                      a radius curve with descriptors (radius fraction, contrast).
* fibre_type        – dominant local fibre orientation (radial vs transverse).

Method variants are benchmarked in bench/markings.py. ``P`` holds tunable params.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d, median_filter

from ..types import NormalizedIris
from .common import (Candidate, components, masked_background, polar_lab, robust_sigma,
                     shape_attrs, unpad, wrap_pad)


def _valid(norm: NormalizedIris, erode: int = 1) -> np.ndarray:
    v = norm.valid.astype(np.uint8)
    if erode:
        v = cv2.erode(v, np.ones((2 * erode + 1, 2 * erode + 1), np.uint8))
    v[:2] = 0  # pupil-edge rows: shading artefacts
    v[-2:] = 0  # limbal ring
    return v.astype(bool)


def _area_limits(norm, lo=0.0008, hi=0.06):
    n = norm.valid.size
    return max(6, int(lo * n)), int(hi * n)


# ----------------------------------------------------------------------------- dark pigment
def dark_pigment(norm: NormalizedIris, method: str = "bgsub", k: float = 3.0, **P) -> list[Candidate]:
    lab = polar_lab(norm)
    v = _valid(norm)
    amin, amax = _area_limits(norm)
    L = lab[..., 0]
    bgL = masked_background(L, v, (9, 41))
    dL = bgL - L
    s = robust_sigma(dL[v]) if v.any() else 1
    if method == "bgsub":
        score = dL / s
        m = score > k
    elif method == "deltaE":
        bga = masked_background(lab[..., 1], v, (9, 41))
        bgb = masked_background(lab[..., 2], v, (9, 41))
        dE = np.sqrt(dL ** 2 + (lab[..., 1] - bga) ** 2 + (lab[..., 2] - bgb) ** 2)
        chroma_gain = np.hypot(lab[..., 1], lab[..., 2]) - np.hypot(bga, bgb)
        se = robust_sigma(dE[v]) if v.any() else 1
        score = dE / se
        m = (score > k) & (dL > 0) & (chroma_gain > -1.0)
    elif method == "mser":
        L8 = np.clip(L * 2.55, 0, 255).astype(np.uint8)
        pad = 16
        Lp = wrap_pad(L8, pad)
        mser = cv2.MSER_create(delta=4, min_area=amin, max_area=amax)
        regions, _ = mser.detectRegions(Lp)
        mp = np.zeros(Lp.shape, np.uint8)
        for r in regions:
            mp[r[:, 1], r[:, 0]] = 1
        m = unpad(mp, pad).astype(bool) & (dL > 0.5 * k * s)
        score = dL / s
    else:
        raise ValueError(method)
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)).astype(bool)
    return components(m, v, amin, amax, "dark_pigmentation", score, shape_attrs)


# ----------------------------------------------------------------------------- white marking
def white_marking(norm: NormalizedIris, method: str = "deltaL", k: float = 3.0, **P) -> list[Candidate]:
    lab = polar_lab(norm)
    v = _valid(norm, erode=2)
    amin, amax = _area_limits(norm)
    L = lab[..., 0]
    chroma = np.hypot(lab[..., 1], lab[..., 2])
    if method == "deltaL":
        bg = masked_background(L, v, (9, 41))
        bgc = masked_background(chroma, v, (9, 41))
        d = L - bg
        s = robust_sigma(d[v]) if v.any() else 1
        score = d / s
        m = (score > k) & (chroma <= bgc + 2)
    elif method == "tophat":
        L8 = np.clip(L * 2.55, 0, 255).astype(np.uint8)
        pad = 12
        th = cv2.morphologyEx(wrap_pad(L8, pad), cv2.MORPH_TOPHAT,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 11)))
        th = unpad(th, pad).astype(np.float32)
        s = robust_sigma(th[v]) if v.any() else 1
        score = (th - np.median(th[v])) / s if v.any() else th
        m = score > k
    else:
        raise ValueError(method)
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)).astype(bool)
    return components(m, v, amin, amax, "white_marking", score, shape_attrs)


# ----------------------------------------------------------------------------- lacuna
def lacuna(norm: NormalizedIris, method: str = "blackhat", k: float = 2.5,
           min_elong: float = 1.2, **P) -> list[Candidate]:
    """Lacunae: radially elongated dark gaps; little chroma change (vs pigment)."""
    lab = polar_lab(norm)
    v = _valid(norm)
    amin, amax = _area_limits(norm, 0.0015, 0.05)
    L = lab[..., 0]
    chroma = np.hypot(lab[..., 1], lab[..., 2])
    if method == "blackhat":
        L8 = np.clip(L * 2.55, 0, 255).astype(np.uint8)
        pad = 16
        # elongated radially: tall (rows) x narrow (cols) structuring element
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 25))
        bh = unpad(cv2.morphologyEx(wrap_pad(L8, pad), cv2.MORPH_BLACKHAT, se), pad).astype(np.float32)
        s = robust_sigma(bh[v]) if v.any() else 1
        score = (bh - np.median(bh[v])) / s if v.any() else bh
        m = score > k
    elif method == "dog":
        x = L.copy()
        x[~v] = np.median(L[v]) if v.any() else 0
        pad = 16
        xp = wrap_pad(x, pad)
        dog = gaussian_filter(xp, (6, 3)) - gaussian_filter(xp, (2, 1.5))  # dark blobs > 0
        dog = unpad(dog, pad)
        s = robust_sigma(dog[v]) if v.any() else 1
        score = dog / s
        m = score > k
    else:
        raise ValueError(method)
    bgc = masked_background(chroma, v, (9, 41))
    m &= (chroma - bgc) < 4  # not strongly pigmented
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)).astype(bool)
    cands = components(m, v, amin, amax, "lacuna", score, shape_attrs)
    return [c for c in cands if c.attrs["elongation_radial"] >= min_elong * 0.5
            and c.attrs["radial_extent"] >= 4]


# ----------------------------------------------------------------------------- furrows
def furrows(norm: NormalizedIris, method: str = "ridge", k: float = 3.0,
            min_arc_deg: float = 25.0, rho_min: float = 0.35, **P) -> list[Candidate]:
    """Contraction furrows = horizontal dark lines in polar space (constant radius)."""
    lab = polar_lab(norm)
    v = _valid(norm)
    L = lab[..., 0].copy()
    L[~v] = np.median(L[v]) if v.any() else 0
    n_r, n_t = L.shape
    pad = 24
    Lp = wrap_pad(L, pad)
    if method == "ridge":
        # smooth along theta, 2nd derivative along r: dark thin line -> positive
        sm = gaussian_filter1d(Lp, 6, axis=1)
        sm = gaussian_filter1d(sm, 1.2, axis=0)
        resp = unpad(np.gradient(np.gradient(sm, axis=0), axis=0), pad)
    elif method == "gabor":
        # Gabor tuned to horizontal stripes (wave along r), long support along theta
        kern = cv2.getGaborKernel((41, 15), sigma=3.0, theta=np.pi / 2, lambd=6.0, gamma=0.25, psi=np.pi)
        kern -= kern.mean()
        resp = unpad(cv2.filter2D(Lp.astype(np.float32), -1, kern), pad)
    else:
        raise ValueError(method)
    s = robust_sigma(resp[v]) if v.any() else 1
    score = (resp - np.median(resp[v])) / s if v.any() else resp
    rows = (np.arange(n_r) + 0.5) / n_r
    m = (score > k) & (rows[:, None] >= rho_min)
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((1, 7), np.uint8)).astype(bool)
    min_cols = int(min_arc_deg / 360 * n_t)
    cands = components(m, v, min_cols, int(0.2 * m.size), "contraction_furrow", score, shape_attrs)
    out = []
    for c in cands:
        if c.attrs["angular_extent"] >= min_cols and c.attrs["radial_extent"] <= max(6, 0.15 * n_r):
            c.attrs["arc_deg"] = round(c.attrs["angular_extent"] / n_t * 360, 1)
            out.append(c)
    return out


# ----------------------------------------------------------------------------- collarette
def collarette(norm: NormalizedIris, method: str = "gradient", search=(0.15, 0.6), **P) -> dict:
    """Estimate the collarette radius curve rho(theta) and descriptive attributes.

    gradient: strongest radial change in smoothed lightness inside `search`.
    texture : strongest change of local texture energy (pupillary zone is smoother /
              differently textured than the ciliary zone).
    Returns {} if not assessable.
    """
    lab = polar_lab(norm)
    v = norm.valid
    n_r, n_t = v.shape
    if v.mean() < 0.3:
        return {}
    L = lab[..., 0].copy()
    L[~v] = np.nan
    col_ok = np.isfinite(L).mean(0) > 0.7
    if col_ok.mean() < 0.3:
        return {}
    Lf = np.where(np.isfinite(L), L, np.nanmedian(L))
    Lf = gaussian_filter1d(gaussian_filter1d(Lf, 8, axis=1, mode="wrap"), 1.5, axis=0)
    r0, r1 = int(search[0] * n_r), int(search[1] * n_r)
    if method == "gradient":
        prof = np.abs(np.gradient(Lf, axis=0))
    elif method == "texture":
        hp = Lf - gaussian_filter1d(Lf, 3, axis=1, mode="wrap")
        hp = np.abs(L - np.where(np.isfinite(L), gaussian_filter1d(np.nan_to_num(L, nan=np.nanmedian(L)), 3, axis=1, mode="wrap"), 0))
        e = gaussian_filter1d(gaussian_filter1d(np.nan_to_num(hp), 6, axis=1, mode="wrap"), 2, axis=0)
        prof = np.abs(np.gradient(e, axis=0))
    else:
        raise ValueError(method)
    band = prof[r0:r1]
    rows = band.argmax(0) + r0
    rows = median_filter(rows.astype(float), size=31, mode="wrap")
    rho = (rows + 0.5) / n_r
    rho_valid = rho[col_ok]
    # descriptors: collarette band vs neighbours (white?), radius fraction (dilated?)
    Lraw = np.where(np.isfinite(L), L, np.nan)
    ring = [Lraw[max(0, int(r) - 1):int(r) + 2, j] for j, r in enumerate(rows)]
    inner = [Lraw[max(0, int(r) - 6):max(1, int(r) - 2), j] for j, r in enumerate(rows)]
    outer = [Lraw[int(r) + 3:int(r) + 7, j] for j, r in enumerate(rows)]
    def nm(xs):
        xs = np.concatenate([x[np.isfinite(x)] for x in xs if x.size]) if xs else np.array([])
        return float(np.median(xs)) if xs.size else float("nan")
    contrast = nm(ring) - 0.5 * (nm(inner) + nm(outer))
    return {
        "rho_curve": rho,
        "radius_frac_median": float(np.median(rho_valid)),
        "radius_frac_iqr": float(np.subtract(*np.percentile(rho_valid, [75, 25]))),
        "contrast_L": round(contrast, 2),
        "coverage": float(col_ok.mean()),
        "method": method,
    }


# ----------------------------------------------------------------------------- fibre type
def fibre_orientation(norm: NormalizedIris, method: str = "structure_tensor", **P) -> dict:
    """Radial fibres appear as vertical streaks in polar space (intensity varies along
    theta) -> gradient energy along theta (x) dominates. Transverse fibres -> gradient
    along r (y) dominates. Returns radial_index in [-1, 1] (+1 = radial) per sector."""
    lab = polar_lab(norm)
    v = _valid(norm)
    if v.mean() < 0.3:
        return {}
    L = lab[..., 0].copy()
    L[~v] = np.median(L[v])
    pad = 8
    Lp = wrap_pad(L, pad)
    if method == "structure_tensor":
        g = gaussian_filter(Lp, 0.8)
        gx = np.gradient(g, axis=1)
        gy = np.gradient(g, axis=0)
        Ex = unpad(gaussian_filter(gx * gx, 3), pad)
        Ey = unpad(gaussian_filter(gy * gy, 3), pad)
    elif method == "gabor":
        def energy(theta):
            e = 0
            for psi in (0, np.pi / 2):
                kern = cv2.getGaborKernel((15, 15), 2.5, theta, 5.0, 0.5, psi)
                kern -= kern.mean()
                e = e + cv2.filter2D(Lp.astype(np.float32), -1, kern) ** 2
            return unpad(e, pad)
        Ex = energy(0.0)          # wave along x (theta)  -> vertical streaks -> radial fibres
        Ey = energy(np.pi / 2)    # wave along y (radius) -> horizontal bands -> transverse
    else:
        raise ValueError(method)
    idx = (Ex - Ey) / (Ex + Ey + 1e-6)
    n_t = idx.shape[1]
    per_sector = {}
    for k in range(12):
        c0 = int(((k - 0.5) % 12) / 12 * n_t)
        cols = (np.arange(n_t // 12) + c0) % n_t
        vv = v[:, cols]
        if vv.mean() > 0.3:
            per_sector[12 if k == 0 else k] = round(float(np.median(idx[:, cols][vv])), 3)
    glob = float(np.median(idx[v]))
    return {"radial_index": round(glob, 3), "per_sector": per_sector, "method": method}


def _solidity(m: np.ndarray) -> float:
    cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 0.0
    c = max(cnts, key=cv2.contourArea)
    hull = cv2.contourArea(cv2.convexHull(c))
    return float(m.sum() / hull) if hull > 0 else 1.0


def strict_filter(cands: list[Candidate], k: float, score_gain: float = 1.25,
                  min_solidity: float = 0.6, max_per_eye: int = 3) -> list[Candidate]:
    """Precision-oriented post-filter: the candidate's MEAN score must be well above the
    detection threshold, the region compact (not a ragged texture fragment), and at most
    `max_per_eye` strongest candidates are kept."""
    keep = [c for c in cands if c.score >= score_gain * k and _solidity(c.mask) >= min_solidity]
    return sorted(keep, key=lambda c: -c.score)[:max_per_eye]


def _with_strict(fn):
    def wrapped(norm, method, k=3.0, strict=False, **P):
        c = fn(norm, method, k=k, **P)
        return strict_filter(c, k) if strict else c
    wrapped.__name__ = fn.__name__
    wrapped.__doc__ = fn.__doc__
    return wrapped


DETECTORS = {
    "dark_pigmentation": (_with_strict(dark_pigment), ("bgsub", "deltaE", "mser")),
    "white_marking": (_with_strict(white_marking), ("deltaL", "tophat")),
    "lacuna": (_with_strict(lacuna), ("blackhat", "dog")),
    "contraction_furrow": (_with_strict(furrows), ("ridge", "gabor")),
}
