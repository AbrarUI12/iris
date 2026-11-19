"""12 radial sectors x 3 concentric rings on the normalised iris.

Sector k (1..12) is centred on clock hour k as seen in the photo (sector 12 spans
11:30-12:30). Rings are fractions of the pupil->limbus distance (default thirds;
practitioner maps may move the inner/middle boundary onto the collarette).

The *meaning* attached to a zone is practitioner-authored configuration
(``labels``), kept separate from the geometry and versioned. No anatomical or health
claim is made by this module.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .geometry import Ellipse

RINGS = ("inner", "middle", "outer")


@dataclass
class ZoneMap:
    version: str = "geometry_v1"
    ring_edges: tuple[float, float, float, float] = (0.0, 1 / 3, 2 / 3, 1.0)
    # labels[eye_side][sector][ring] -> practitioner label (free text, optional)
    labels: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> "ZoneMap":
        d = json.loads(Path(path).read_text())
        return cls(d["version"], tuple(d.get("ring_edges", (0, 1 / 3, 2 / 3, 1))), d.get("labels", {}))

    # ---------------------------------------------------------------- polar indexing
    @staticmethod
    def sector_of_col(col: np.ndarray | int, n_t: int) -> np.ndarray:
        """Column -> sector 1..12 (column 0 = 12 o'clock, clockwise)."""
        hours = (np.asarray(col) + 0.5) / n_t * 12.0  # 0..12, 0 = 12 o'clock
        k = np.floor(hours + 0.5).astype(int) % 12
        return np.where(k == 0, 12, k)

    def ring_of_row(self, row: np.ndarray | int, n_r: int) -> np.ndarray:
        rho = (np.asarray(row) + 0.5) / n_r
        idx = np.searchsorted(np.asarray(self.ring_edges[1:-1]), rho, side="right")
        return np.asarray(RINGS, dtype=object)[idx]

    def zone_of_polar(self, row: float, col: float, n_r: int, n_t: int) -> tuple[int, str]:
        return int(self.sector_of_col(int(col), n_t)), str(self.ring_of_row(int(row), n_r))

    def label(self, eye: str, sector: int, ring: str) -> str | None:
        return self.labels.get(eye, {}).get(str(sector), {}).get(ring)

    # ---------------------------------------------------------------- overlay
    def draw(self, bgr: np.ndarray, pupil: Ellipse, iris: Ellipse, color=(0, 255, 255)) -> np.ndarray:
        from .normalize import polar_maps
        v = bgr.copy()
        n_t = 360
        for edge in self.ring_edges:
            mx, my = polar_maps(pupil, iris, 1, n_t)
            rho = edge
            px, py = pupil.point_at(np.arange(n_t) / n_t * 2 * np.pi - np.pi / 2)
            ix, iy = iris.point_at(np.arange(n_t) / n_t * 2 * np.pi - np.pi / 2)
            pts = np.stack([(1 - rho) * px + rho * ix, (1 - rho) * py + rho * iy], 1)
            cv2.polylines(v, [np.round(pts).astype(np.int32)], True, color, 1, cv2.LINE_AA)
        for k in range(12):
            phi = (k + 0.5) / 12 * 2 * np.pi - np.pi / 2  # sector boundaries at half hours
            p0 = pupil.point_at(np.array([phi]))
            p1 = iris.point_at(np.array([phi]))
            cv2.line(v, (int(p0[0][0]), int(p0[1][0])), (int(p1[0][0]), int(p1[1][0])), color, 1, cv2.LINE_AA)
        for k in range(1, 13):
            phi = k / 12 * 2 * np.pi - np.pi / 2
            x, y = iris.point_at(np.array([phi]))
            x = iris.cx + (x[0] - iris.cx) * 1.12
            y = iris.cy + (y[0] - iris.cy) * 1.12
            cv2.putText(v, str(k), (int(x) - 5, int(y) + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
        return v
