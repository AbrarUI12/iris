"""Internal result containers (dataclasses) and the external JSON schema (pydantic)."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal, Optional

import numpy as np
from pydantic import BaseModel, Field

from .geometry import Ellipse


# ----------------------------------------------------------------------------- internal
@dataclass
class SegResult:
    """Output of every segmenter.

    pupil/iris: boundary ellipses (image pixel coords). None => failed.
    iris_mask: visible, noise-free iris pixels (annulus minus eyelids/lashes/glints)
    pupil_mask: pupil region
    """

    pupil: Optional[Ellipse]
    iris: Optional[Ellipse]
    iris_mask: Optional[np.ndarray] = None
    pupil_mask: Optional[np.ndarray] = None
    method: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.pupil is not None and self.iris is not None


@dataclass
class NormalizedIris:
    """Rubber-sheet (Daugman) normalised iris.

    image: H_r x W_theta x 3 (BGR, uint8). Row 0 = pupil boundary, last row = limbus.
    Column 0 = angle 0 = 12 o'clock, increasing clockwise *as the viewer sees the photo*
    for the right eye, and mirrored for the left eye so the same column index means the
    same nasal/temporal anatomy (configurable in zones).
    valid: H_r x W_theta bool, False where occluded / glint / outside image.
    map_x/map_y: source pixel coordinates of each polar sample (for back-projection).
    """

    image: np.ndarray
    valid: np.ndarray
    map_x: np.ndarray
    map_y: np.ndarray
    eye_side: str = "right"


# ----------------------------------------------------------------------------- external
class Status(str, Enum):
    present_candidate = "present_candidate"
    not_detected = "not_detected_in_assessable_image"
    unassessable = "unassessable"


class QualityOut(BaseModel):
    status: Literal["accepted", "retake", "review"]
    reasons: list[str] = []
    metrics: dict[str, float] = {}
    color_assessable: bool = True
    fine_markings_assessable: bool = True


class EllipseOut(BaseModel):
    cx: float
    cy: float
    a: float
    b: float
    theta: float


class SegmentationOut(BaseModel):
    method_version: str
    pupil: Optional[EllipseOut]
    iris: Optional[EllipseOut]
    visible_iris_fraction: float


class ColorOut(BaseModel):
    category: Literal["blue", "brown", "green_mixed", "uncertain"]
    confidence: float
    lab_median: list[float]
    method_version: str
    review_status: Literal["unreviewed", "confirmed", "corrected"] = "unreviewed"


class MarkingOut(BaseModel):
    marking_type: Literal[
        "lacuna", "dark_pigmentation", "white_marking", "collarette",
        "contraction_furrow", "fibre_type",
    ]
    status: Status
    region_id: Optional[str] = None
    sector: Optional[int] = Field(default=None, ge=1, le=12)
    ring: Optional[Literal["inner", "middle", "outer"]] = None
    polar_bbox: Optional[list[int]] = None  # r0, t0, r1, t1 in normalized coords
    score: Optional[float] = None
    attributes: dict = {}
    method_version: str = ""
    review_status: Literal["candidate_needs_review", "confirmed", "rejected"] = "candidate_needs_review"


class EyeObservation(BaseModel):
    image_id: str
    eye_side: Literal["left", "right"]
    quality: QualityOut
    segmentation: Optional[SegmentationOut] = None
    color: Optional[ColorOut] = None
    markings: list[MarkingOut] = []
    timings_ms: dict[str, float] = {}


class ScanObservation(BaseModel):
    scan_id: str
    pipeline_version: str
    zone_map_version: str
    config: dict
    eyes: list[EyeObservation]
    disclaimer: str = (
        "Descriptive image observations only. Iris colour, zone and marking outputs are "
        "appearance/geometry measurements and are not validated indicators of health."
    )
