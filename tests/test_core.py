import json

import numpy as np
import pytest

from aeye_iris.geometry import Ellipse, ellipse_from_mask, ransac_ellipse
from aeye_iris.normalize import normalize, polar_maps
from aeye_iris.types import SegResult
from aeye_iris.zones import ZoneMap
from synth.eye import render_eye


def test_ellipse_mask_roundtrip():
    e = Ellipse(100, 80, 40, 30, 20)
    f = ellipse_from_mask(e.to_mask((200, 220)))
    assert abs(f.cx - 100) < 1 and abs(f.cy - 80) < 1
    assert abs(max(f.a, f.b) - 40) < 1.5 and abs(min(f.a, f.b) - 30) < 1.5


def test_ransac_rejects_outliers():
    rng = np.random.default_rng(0)
    phi = rng.uniform(0, 2 * np.pi, 80)
    xs, ys = 50 + 20 * np.cos(phi), 60 + 20 * np.sin(phi)
    xs[:15] += rng.uniform(5, 20, 15)
    e = ransac_ellipse(xs, ys, tol=1.0)
    assert abs(e.cx - 50) < 1 and abs(e.r - 20) < 1


@pytest.mark.parametrize("col,expected", [(0, 12), (511, 12), (20, 12), (22, 1), (64, 2), (128, 3), (256, 6), (384, 9)])
def test_sector_clock_convention(col, expected):
    # 512 columns; sector 12 spans 11:30-12:30 (cols -21.3..21.3), 3 o'clock at col 128
    assert ZoneMap.sector_of_col(col, 512) == expected


def test_rings():
    z = ZoneMap()
    assert z.ring_of_row(0, 64) == "inner"
    assert z.ring_of_row(32, 64) == "middle"
    assert z.ring_of_row(63, 64) == "outer"


def test_polar_maps_orientation():
    p, i = Ellipse.circle(100, 100, 20), Ellipse.circle(100, 100, 60)
    mx, my = polar_maps(p, i, 8, 360)
    # column 0 = 12 o'clock: straight up (smaller y)
    assert abs(mx[-1, 0] - 100) < 1 and my[-1, 0] < 50
    # column 90 = 3 o'clock (right side of photo)
    assert mx[-1, 90] > 150 and abs(my[-1, 90] - 100) < 1
    # left eye mirrored
    mx2, _ = polar_maps(p, i, 8, 360, eye_side="left", mirror_left=True)
    assert mx2[-1, 90] < 50


def test_normalize_valid_fraction():
    img, gt = render_eye(seed=3)
    seg = SegResult(gt["pupil"], gt["iris"], gt["iris_mask"], gt["pupil_mask"])
    n = normalize(img, seg, 32, 256)
    assert n.image.shape == (32, 256, 3)
    assert 0.6 < n.valid.mean() <= 1.0


@pytest.mark.parametrize("name", ["daugman", "hough", "raycast"])
def test_classical_segmenters_run(name):
    from aeye_iris.segment import build
    img, gt = render_eye(seed=4)
    r = build(name)(img)
    assert r.ok
    assert abs(r.iris.cx - gt["iris"].cx) < 0.15 * gt["iris"].r


def test_fibre_orientation_sign():
    from aeye_iris.markings.detectors import fibre_orientation
    for fibre, sign in (("radial", 1), ("transverse", -1)):
        img, gt = render_eye(seed=5, fibre=fibre, glare=False)
        n = normalize(img, SegResult(gt["pupil"], gt["iris"], gt["iris_mask"], gt["pupil_mask"]))
        assert np.sign(fibre_orientation(n)["radial_index"]) == sign


@pytest.mark.parametrize("kind", ["dark_pigmentation", "white_marking", "lacuna", "contraction_furrow"])
def test_injected_marking_detected(kind):
    from aeye_iris.markings.detectors import DETECTORS
    from bench.metrics import iou
    from synth.markings import inject
    img, gt = render_eye(seed=6, glare=False, lid_cover=0.0)
    n = normalize(img, SegResult(gt["pupil"], gt["iris"], gt["iris_mask"], gt["pupil_mask"]))
    rng = np.random.default_rng(1)
    n.image, gmask, _ = inject(n.image, n.valid, kind, rng, strength=1.0)
    fn, methods = DETECTORS[kind]
    cands = fn(n, methods[0], k=2.5)
    assert max((iou(c.mask, gmask) for c in cands), default=0) > 0.1


def test_pipeline_schema_and_determinism(tmp_path):
    from aeye_iris.pipeline import Pipeline
    a, _ = render_eye(seed=4)
    b, _ = render_eye(seed=7, color="blue")
    p = Pipeline()
    o1 = p.analyze({"left": a, "right": b}).model_dump(exclude={"eyes": {"__all__": {"timings_ms"}}})
    o2 = p.analyze({"left": a, "right": b}).model_dump(exclude={"eyes": {"__all__": {"timings_ms"}}})
    assert json.dumps(o1, sort_keys=True, default=str) == json.dumps(o2, sort_keys=True, default=str)
    assert {e["eye_side"] for e in o1["eyes"]} == {"left", "right"}
    for e in o1["eyes"]:
        kinds = {m["marking_type"] for m in e["markings"]}
        assert kinds == {"dark_pigmentation", "white_marking", "lacuna", "contraction_furrow",
                         "collarette", "fibre_type"}


def test_auto_segmenter_selects_by_framing():
    from aeye_iris.segment import build
    seg = build("auto", model="seg_small_160.onnx")
    img, gt = render_eye(seed=4)
    r = seg(img)
    assert r.ok and r.method.startswith("auto->")
    assert "closeup_frac" in r.extra


# ---- real-photo failure modes (dark iris read as pupil; ring artefacts) -------------
def _dark_eye(seed=11):
    import synth.eye as E
    E.IRIS_BGR["very_dark"] = (22, 28, 38)  # dark brown iris, pupil barely separable
    return render_eye(seed=seed, color="very_dark", glare=False)


def test_plausibility_accepts_correct_segmentation():
    from aeye_iris.plausibility import measures, reasons
    for seed, color in ((1, "brown"), (3, "blue"), (5, "green_mixed")):
        img, gt = render_eye(seed=seed, color=color, glare=False)
        seg = SegResult(gt["pupil"], gt["iris"], gt["iris_mask"], gt["pupil_mask"])
        assert reasons(measures(img, seg)) == [], (seed, color)


def test_plausibility_rejects_iris_taken_for_pupil():
    """User photo failure: the whole dark iris was returned as the 'pupil' and a circle
    ~3x larger (sclera, lids, skin) as the 'iris'."""
    from aeye_iris.geometry import Ellipse
    from aeye_iris.plausibility import measures, reasons
    img, gt = _dark_eye()
    true_iris = gt["iris"]
    wrong_iris = Ellipse.circle(true_iris.cx, true_iris.cy, true_iris.r * 2.8)
    seg = SegResult(Ellipse.circle(true_iris.cx, true_iris.cy, true_iris.r), wrong_iris,
                    wrong_iris.to_mask(img.shape[:2]), None)
    assert reasons(measures(img, seg)), "implausible iris outline must trigger a retake"


def test_assess_retakes_out_of_frame_iris():
    from aeye_iris.geometry import Ellipse
    from aeye_iris.quality import assess
    img, gt = render_eye(seed=2)
    off = Ellipse.circle(-400, 150, 900)
    q = assess(img, SegResult(gt["pupil"], off, off.to_mask(img.shape[:2]), gt["pupil_mask"]))
    assert q.status == "retake" and not q.color_assessable
    assert any("iris not found" in r for r in q.reasons)


def test_no_full_ring_marking_candidates():
    from aeye_iris.markings.detectors import DETECTORS
    img, gt = render_eye(seed=6, glare=False, lid_cover=0.0)
    n = normalize(img, SegResult(gt["pupil"], gt["iris"], gt["iris_mask"], gt["pupil_mask"]))
    n.image = n.image.copy()
    n.image[40:42, :] = (n.image[40:42, :] * 0.4).astype(n.image.dtype)  # full 360-degree dark ring
    n.image[10:12, :] = np.clip(n.image[10:12, :].astype(int) + 90, 0, 255).astype(n.image.dtype)  # bright ring
    for kind, (fn, methods) in DETECTORS.items():
        for c in fn(n, methods[0], k=2.5):
            assert c.attrs.get("angular_extent", 0) < 0.85 * n.valid.shape[1], (kind, c.attrs)
