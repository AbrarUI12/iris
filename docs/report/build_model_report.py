"""Build docs/A-EYE_Iris_Model_Report.pdf (HTML -> PDF via WeasyPrint).

python docs/report/build_model_report.py
Figures are generated from synthetic eyes (no personal data) and committed benchmark outputs.
"""
from __future__ import annotations

import base64
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from aeye_iris.normalize import normalize  # noqa: E402
from aeye_iris.segment import build  # noqa: E402
from aeye_iris.zones import ZoneMap  # noqa: E402
from synth.eye import render_eye  # noqa: E402

OUT = ROOT / "docs" / "A-EYE_Iris_Model_Report.pdf"


def b64_img(arr_or_path, fmt=".png", max_w=None) -> str:
    if isinstance(arr_or_path, (str, Path)):
        arr = cv2.imread(str(arr_or_path))
    else:
        arr = arr_or_path
    if max_w and arr.shape[1] > max_w:
        s = max_w / arr.shape[1]
        arr = cv2.resize(arr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    ok, enc = cv2.imencode(fmt, arr, [cv2.IMWRITE_JPEG_QUALITY, 88] if fmt == ".jpg" else [])
    mime = "jpeg" if fmt == ".jpg" else "png"
    return f"data:image/{mime};base64," + base64.b64encode(enc.tobytes()).decode()


def pipeline_figure() -> tuple[str, str]:
    """Synthetic eye -> segmentation overlay + zone grid, and its unrolled polar image."""
    img, _ = render_eye(seed=4, color="brown", glare=True)
    seg = build("daugman", channel="red", clahe=True, inpaint=True)(img)
    v = ZoneMap().draw(img, seg.pupil, seg.iris)
    for e, c in ((seg.iris, (0, 200, 0)), (seg.pupil, (255, 0, 255))):
        cv2.ellipse(v, (int(e.cx), int(e.cy)), (int(e.a), int(e.b)), e.theta, 0, 360, c, 2)
    n = normalize(img, seg, 64, 512)
    polar = cv2.resize(n.image, (512, 128), interpolation=cv2.INTER_NEAREST)
    for k in range(1, 12):
        x = int((k - 0.5) / 12 * 512)
        cv2.line(polar, (x, 0), (x, 127), (0, 255, 255), 1)
    for f in (1 / 3, 2 / 3):
        cv2.line(polar, (0, int(f * 128)), (511, int(f * 128)), (0, 255, 255), 1)
    return b64_img(v), b64_img(polar)


PIPE_SVG = """
<svg viewBox="0 0 760 150" xmlns="http://www.w3.org/2000/svg" font-family="Helvetica" font-size="10.5">
<defs><marker id="a" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="#555"/></marker></defs>
<g fill="#eef3fb" stroke="#4C78A8" stroke-width="1.2">
<rect x="5" y="20" width="88" height="44" rx="6"/><rect x="113" y="20" width="88" height="44" rx="6"/>
<rect x="221" y="20" width="98" height="44" rx="6"/><rect x="339" y="20" width="98" height="44" rx="6"/>
<rect x="457" y="20" width="88" height="44" rx="6"/><rect x="565" y="20" width="88" height="44" rx="6"/>
<rect x="673" y="20" width="82" height="44" rx="6"/>
<rect x="339" y="92" width="98" height="44" rx="6"/><rect x="457" y="92" width="88" height="44" rx="6"/><rect x="565" y="92" width="88" height="44" rx="6"/>
</g>
<g text-anchor="middle" fill="#111">
<text x="49" y="38">2 photos</text><text x="49" y="52">left + right eye</text>
<text x="157" y="38">1. Quick</text><text x="157" y="52">quality check</text>
<text x="270" y="38">2. Find iris</text><text x="270" y="52">+ pupil (auto)</text>
<text x="388" y="38">3. Plausibility +</text><text x="388" y="52">quality gate</text>
<text x="501" y="38">4. Unroll iris</text><text x="501" y="52">64 x 512</text>
<text x="609" y="38">5. 12 sectors</text><text x="609" y="52">x 3 rings</text>
<text x="714" y="38">JSON +</text><text x="714" y="52">overlays</text>
<text x="388" y="110">retake with</text><text x="388" y="124">reason</text>
<text x="501" y="110">6. Colour</text><text x="501" y="124">blue/brown/mixed</text>
<text x="609" y="110">7. Six marking</text><text x="609" y="124">candidates</text>
</g>
<g stroke="#555" stroke-width="1.2" marker-end="url(#a)" fill="none">
<line x1="93" y1="42" x2="111" y2="42"/><line x1="201" y1="42" x2="219" y2="42"/><line x1="319" y1="42" x2="337" y2="42"/>
<line x1="437" y1="42" x2="455" y2="42"/><line x1="545" y1="42" x2="563" y2="42"/><line x1="653" y1="42" x2="671" y2="42"/>
<line x1="388" y1="64" x2="388" y2="90"/><line x1="501" y1="64" x2="501" y2="90"/><line x1="609" y1="64" x2="609" y2="90"/>
<path d="M653,114 C700,114 714,90 714,66"/>
</g></svg>
"""


def html() -> str:
    eye, polar = pipeline_figure()
    pareto = b64_img(ROOT / "results/seg_pareto.png", max_w=1100)
    conf = b64_img(ROOT / "results/color/confusion.png", max_w=700)
    upol = ROOT / "results/visual_test_auto/sheet_00.jpg"
    upol_img = b64_img(cv2.imread(str(upol))[:360], ".jpg", max_w=1400) if upol.exists() else ""
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>A-EYE Iris Model Report</title>
<style>
@page {{ size: A4; margin: 17mm 16mm 18mm 16mm;
  @bottom-center {{ content: "A-EYE iris model report  ·  page " counter(page) " of " counter(pages); font-size: 8pt; color: #777; }} }}
body {{ font-family: Helvetica, Arial, sans-serif; font-size: 9.6pt; line-height: 1.42; color: #1d1d1f; }}
h1 {{ font-size: 21pt; margin: 0 0 4pt; color: #1f3b63; }}
h2 {{ font-size: 13.5pt; color: #1f3b63; border-bottom: 1.5px solid #c9d6ea; padding-bottom: 2pt; margin: 16pt 0 6pt; page-break-after: avoid; }}
h3 {{ font-size: 10.8pt; margin: 10pt 0 3pt; color: #2b4f80; page-break-after: avoid; }}
p {{ margin: 3pt 0 5pt; }} ul {{ margin: 2pt 0 6pt 14pt; padding: 0; }} li {{ margin: 1pt 0; }}
table {{ border-collapse: collapse; width: 100%; margin: 4pt 0 8pt; font-size: 8.6pt; page-break-inside: avoid; }}
th {{ background: #eaf0f8; text-align: left; padding: 3pt 4pt; border-bottom: 1px solid #b9c7dc; }}
td {{ padding: 2.6pt 4pt; border-bottom: 1px solid #e3e8f0; vertical-align: top; }}
tr.win td {{ background: #eef8ee; font-weight: bold; }}
.sub {{ color: #555; font-size: 10pt; margin-bottom: 8pt; }}
.box {{ border: 1px solid #c9d6ea; background: #f6f9fd; padding: 6pt 9pt; border-radius: 4pt; margin: 6pt 0 8pt; page-break-inside: avoid; }}
.warn {{ border-color: #e6c07b; background: #fff8ea; }}
.kpis {{ display: flex; gap: 6pt; margin: 6pt 0 8pt; }}
.kpi {{ flex: 1; border: 1px solid #c9d6ea; border-radius: 4pt; padding: 5pt 6pt; background: #fbfcfe; }}
.kpi b {{ display: block; font-size: 15pt; color: #1f3b63; }} .kpi span {{ font-size: 8pt; color: #555; }}
.fig {{ text-align: center; margin: 4pt 0 8pt; page-break-inside: avoid; }} .fig img {{ max-width: 100%; }}
.cap {{ font-size: 8pt; color: #555; margin-top: 2pt; }}
.two {{ display: flex; gap: 10pt; align-items: flex-start; }} .two > div {{ flex: 1; }}
code {{ font-family: Courier, monospace; font-size: 8.6pt; background: #f1f3f6; padding: 0 2pt; }}
</style></head><body>

<h1>A-EYE Iris Model</h1>
<div class="sub">How it works · what data it was built on · how well it performs<br>
Repository <code>AbrarUI12/iris</code>, branch <code>claude/tender-cray-w7opo5</code> · 24 November 2026</div>

<div class="kpis">
<div class="kpi"><b>28 / 30</b><span>real colour iris close-ups correctly outlined (visual check)</span></div>
<div class="kpi"><b>0.91</b><span>iris Dice on real infrared eyes (340 held-out photos)</span></div>
<div class="kpi"><b>0.88</b><span>colour accuracy (blue / brown / green-mixed)</span></div>
<div class="kpi"><b>0.5 s</b><span>full two-eye analysis on 1 CPU core (SRS limit 30 s)</span></div>
<div class="kpi"><b>3.8 MB</b><span>only learned image model; no GPU, no LLM</span></div>
</div>

<div class="box"><b>In one paragraph.</b> The system takes one photo of each eye. It checks photo quality, finds the
pupil and iris outline, rejects photos where the outline is implausible, unrolls the iris into a rectangle and
divides it into the SRS's 12 sectors × 3 rings. It then measures iris colour and looks for candidates of the six SRS
markings. The output is a structured JSON record plus overlay images. Everything is a <i>descriptive image
measurement</i>: the system does not infer organ function or health, and every marking is a "candidate" for practitioner review.</div>

<h2>1. How the model works</h2>
<div class="fig">{PIPE_SVG}<div class="cap">Figure 1 – Processing chain, run separately for each eye.</div></div>

<div class="two">
<div class="fig"><img src="{eye}" style="max-height:170pt"><div class="cap">Figure 2a – Iris (green) and pupil (magenta) outlines with the 12-sector × 3-ring grid (synthetic example eye).</div></div>
<div class="fig"><img src="{polar}" style="margin-top:30pt"><div class="cap">Figure 2b – The same iris unrolled to 64 × 512: left→right = clock face from 12 o'clock clockwise, top→bottom = pupil edge → outer iris edge.</div></div>
</div>

<table>
<tr><th style="width:22%">Step</th><th>What happens</th><th style="width:24%">Tools</th></tr>
<tr><td>1. Quick quality check (~5 ms)</td><td>Sharpness (variance of the Laplacian), edge strength, brightness and clipping on a 320 px copy. A clearly bad photo is sent back with a reason.</td><td>OpenCV, NumPy</td></tr>
<tr><td>2. Find pupil and iris – <b>auto</b></td><td>The classical <b>Daugman</b> search runs first: it averages brightness along candidate circles and keeps the strongest dark-to-light edge, checked on both side arcs. A second test decides which boundary is pupil and which is iris. If the iris fills ≥ 40% of the frame (close-up), that result is kept. Otherwise a small <b>neural network</b> labels every pixel as background / iris / pupil, and ellipses are fitted to its output. Eyelids, lashes and reflections are then masked out by colour (CIELAB) and brightness.</td><td>OpenCV, NumPy, SciPy; ONNX Runtime for the network</td></tr>
<tr><td>3. Plausibility + quality gate</td><td>Retake if the outline does not look like an eye: iris centre outside the photo, outline not round, "iris" ring not darker than the white of the eye, mostly sclera inside the ring, or large reflections. Then a small learned score (13 measurements incl. JPEG blockiness) decides accept / review / retake.</td><td>NumPy; logistic regression (scikit-learn for training)</td></tr>
<tr><td>4. Unroll ("rubber sheet")</td><td>Each point is placed at a fixed fraction between the pupil edge and the iris edge, so pupil size does not move the zones.</td><td>OpenCV <code>remap</code></td></tr>
<tr><td>5. Zones</td><td>Column → sector 1–12 (centred on clock hours), row → inner / middle / outer ring. Practitioner zone labels live in a separate, versioned map file.</td><td>NumPy</td></tr>
<tr><td>6. Colour</td><td>Median CIELAB colour of the visible, glare-free iris pixels from the original photo (5 numbers) → logistic model → blue / brown / green-mixed, or "uncertain" when the top two classes are close.</td><td>OpenCV, NumPy</td></tr>
<tr><td>7. Markings</td><td><b>Dark pigmentation</b>: darker and more coloured than the local background. <b>White marking</b>: lighter and less saturated, reflections excluded. <b>Lacuna</b>: radially elongated dark gap without colour change. <b>Contraction furrow</b>: thin dark arc following the iris circle. <b>Collarette</b>: radius where the texture changes. <b>Fibre type</b>: dominant fibre direction (radial vs transverse) per sector. Each candidate gets a sector, ring, score and "needs review" status.</td><td>OpenCV, SciPy filters (median, difference-of-Gaussians, ridge, structure tensor)</td></tr>
<tr><td>Output</td><td>Typed <code>ScanObservation</code> JSON with both eyes, quality reasons, outlines, colour, markings, method versions, timings and a disclaimer, plus overlay PNGs.</td><td>Pydantic</td></tr>
</table>

<h3>The learned segmentation network</h3>
<table>
<tr><th>Item</th><th>Value</th></tr>
<tr><td>Architecture</td><td>MobileNetV3-small encoder (ImageNet-pretrained) + small U-Net-style decoder (32 channels, depthwise-separable convs); 3 classes + edge head for training</td></tr>
<tr><td>Size / input</td><td>0.96 M parameters, 3.8 MB ONNX, 160 × 160 RGB input</td></tr>
<tr><td>Training</td><td>15 epochs (best epoch 14), batch 16, AdamW (weight decay 1e-4), peak LR 2e-3 decoder / 6e-4 encoder, one-cycle schedule</td></tr>
<tr><td>Loss</td><td>Weighted cross-entropy (0.5 / 1.0 / 1.5) + soft Dice for iris and pupil + 0.5 × edge loss</td></tr>
<tr><td>Augmentation</td><td>Iris-centred zoom (so close-ups are covered), rotation ±15°, flips, lighting and white-balance changes, blur, noise, JPEG, lens vignette</td></tr>
<tr><td>Training data</td><td>1,729 images (931 synthetic colour + 798 real infrared); 365 for validation; split by person</td></tr>
<tr><td>Hardware</td><td>4 CPU cores, ~48 s per epoch (≈ 12 minutes in total)</td></tr>
</table>

<h2>2. Datasets used</h2>
<table>
<tr><th style="width:18%">Dataset</th><th style="width:22%">What it is</th><th style="width:21%">What it provides</th><th style="width:13%">Split (train / val / test)</th><th>Licence / note</th></tr>
<tr><td><b>Synthetic Human Eyes</b> (Kaggle)</td><td>Rendered colour periocular images, 640 × 360</td><td>Exact iris / pupil / sclera masks; eyelids, gaze, many iris colours. 1,573 kept after mask QC</td><td>931 / 228 / 414</td><td>CC BY 4.0. About 1 in 10 source masks still mismatch the image</td></tr>
<tr><td><b>Pupil-Eye-Iris NIR</b> (Kaggle)</td><td>Real grayscale infrared eye photos, 640 × 480, 85 people</td><td>Human-drawn iris, pupil and eye masks: the only real-eye masks available</td><td>798 / 137 / 340</td><td>Listed MIT; original photo source undocumented (to verify)</td></tr>
<tr><td><b>UPOL Iris Database</b></td><td>Real colour iris close-ups, 576 × 768, 64 people × 2 eyes × 3</td><td>Iridology-style photos with visible texture. No masks → visual grading, hand colour labels (128 eyes), hosts for marking tests, self-training</td><td>198 / 60 / 126</td><td>Academic use only</td></tr>
<tr><td>UBIRIS.v2 (Kaggle mirror)</td><td>Visible-light eye photos</td><td>Only its 2,250 masks were readable</td><td>not used</td><td>Images password-protected by the authors' licence</td></tr>
<tr><td>Periorbital dataset (Zenodo)</td><td>2,842 two-eye colour crops, dark irises</td><td>Iris / sclera / lid masks (no pupil label)</td><td>not used</td><td>Rejected: photos are research-only</td></tr>
</table>
<p>All splits are by <b>person</b>, so the same eye never appears in both training and test. Settings were chosen on
validation data; every number below is on test data that was never used for tuning.</p>

<h2>3. How the model performs</h2>
<h3>3.1 Finding the iris and pupil (segmentation)</h3>
<p><b>Dice</b> = overlap between predicted and true mask (1.0 = perfect); <b>fail</b> = share of images with Dice below 0.8;
time = median per eye on one CPU thread.</p>
<table>
<tr><th>Method</th><th>Colour synthetic<br>iris Dice</th><th>fail</th><th>Real infrared<br>iris Dice</th><th>fail</th><th>pupil Dice<br>(infrared)</th><th>ms / eye</th><th>Model size</th></tr>
<tr class="win"><td>auto (shipped default)</td><td>0.778</td><td>30%</td><td>0.912</td><td>7%</td><td>0.917</td><td>29–52</td><td>3.8 MB</td></tr>
<tr><td>Learned small, 160 px</td><td>0.795</td><td>28%</td><td>0.917</td><td>6%</td><td>0.918</td><td>8–12</td><td>3.8 MB</td></tr>
<tr><td>Learned small, 224 px</td><td>0.788</td><td>29%</td><td>0.899</td><td>17%</td><td>0.927</td><td>13–16</td><td>3.8 MB</td></tr>
<tr><td>Learned small, 320 px</td><td>0.789</td><td>29%</td><td>0.907</td><td>13%</td><td>0.925</td><td>21–27</td><td>3.8 MB</td></tr>
<tr><td>Learned large, 224 px</td><td>0.793</td><td>29%</td><td>0.913</td><td>14%</td><td>0.927</td><td>17–22</td><td>12.1 MB</td></tr>
<tr><td>Learned small 160, INT8</td><td>0.534</td><td>95%</td><td>0.640</td><td>86%</td><td>0.734</td><td>10–18</td><td>1.3 MB</td></tr>
<tr><td>Self-trained small 160 (candidate)</td><td>0.788</td><td>29%</td><td>0.895</td><td>14%</td><td>0.919</td><td>12–23</td><td>3.8 MB</td></tr>
<tr><td>Daugman (best classical)</td><td>0.608</td><td>69%</td><td>0.571</td><td>70%</td><td>0.756</td><td>16–38</td><td>–</td></tr>
<tr><td>Hough circles</td><td>0.459</td><td>91%</td><td>0.454</td><td>74%</td><td>0.675</td><td>32–40</td><td>–</td></tr>
<tr><td>MediaPipe iris (centre only)</td><td>0.537</td><td>83%</td><td>0.428</td><td>77%</td><td>0.684</td><td>19–43</td><td>2.6 MB</td></tr>
</table>
<div class="fig"><img src="{pareto}" style="max-width:82%"><div class="cap">Figure 3 – Accuracy vs speed for every configuration tested (colour synthetic test set).</div></div>

<p><b>Real colour close-ups (UPOL, 30 held-out photos, judged by eye – no masks exist):</b></p>
<table>
<tr><th>Method</th><th>Correct outlines</th><th>Comment</th></tr>
<tr class="win"><td>auto (shipped)</td><td>28 / 30</td><td>Uses Daugman for these close-ups</td></tr>
<tr><td>Daugman</td><td>28 / 30</td><td>Two failures (#7 wrong circle, #22 oversized pupil)</td></tr>
<tr><td>Self-trained learned model</td><td>29 / 30</td><td>Better here, but lost 0.018 Dice on infrared → kept as candidate</td></tr>
<tr><td>Learned model (not self-trained)</td><td>0 / 30</td><td>Never saw real iris texture filling the frame</td></tr>
</table>
{f'<div class="fig"><img src="{upol_img}"><div class="cap">Figure 4 – auto on held-out UPOL close-ups (first two rows of the 30-image test sheet): green = iris outline, magenta = pupil.</div></div>' if upol_img else ''}

<h3>3.2 Photo quality gate</h3>
<p>Tested with 100 good test photos plus 1,400 deliberately damaged copies (blur, motion, dark, overexposed, glare,
JPEG, low resolution). Positive = "should be retaken".</p>
<table>
<tr><th>Gate</th><th>Precision</th><th>Recall</th><th>F1</th><th>Good photos wrongly rejected</th><th>Bad photos let through</th></tr>
<tr><td>Fixed rules</td><td>0.99</td><td>0.57</td><td>0.72</td><td>9 / 100</td><td>599 / 1,400</td></tr>
<tr><td>Calibrated rules</td><td>0.98</td><td>0.73</td><td>0.84</td><td>16 / 100</td><td>380 / 1,400</td></tr>
<tr class="win"><td>Learned gate (shipped)</td><td>0.99</td><td>0.94</td><td>0.97</td><td>9 / 100</td><td>78 / 1,400</td></tr>
</table>
<p>JPEG damage detection rose from 9% to 100% after adding the blockiness measure; AUC 0.98. Precision is inflated by
the 14:1 damaged-to-good ratio; the false-retake rate is the more realistic number.</p>

<h3>3.3 Iris colour</h3>
<div class="fig" style="float:right; width:42%; margin:0 0 4pt 8pt"><img src="{conf}"><div class="cap">Figure 5 – Colour confusion matrix.</div></div>
<div><table style="width:56%">
<tr><th>Class</th><th>Photos</th><th>Precision</th><th>Recall</th><th>F1</th></tr>
<tr><td>Blue</td><td>78</td><td>0.82</td><td>0.90</td><td>0.86</td></tr>
<tr><td>Brown</td><td>150</td><td>0.92</td><td>0.95</td><td>0.93</td></tr>
<tr><td>Green / mixed</td><td>138</td><td>0.88</td><td>0.80</td><td>0.84</td></tr>
<tr class="win"><td>Overall</td><td>366</td><td colspan="3">accuracy 0.88 · macro-F1 0.88</td></tr>
</table>
<p>366 hand-labelled UPOL photos, predicted with 5-fold cross-validation grouped by person. Blue and brown are never
confused; all errors involve green/mixed. Same-eye stability: 97%. Hand-set colour rules reached only 0.52 macro-F1.</p></div>
<div style="clear:both"></div>

<h3>3.4 The six markings</h3>
<div class="box warn">No public dataset labels the SRS markings. These numbers come from markings <b>painted into real
iris photos</b> with a known position, so they measure detectability, not agreement with a practitioner.</div>
<table>
<tr><th>Marking</th><th>Method</th><th>Precision</th><th>Recall</th><th>F1</th><th>False candidates per clean eye</th><th>Recall at half contrast</th></tr>
<tr><td>White marking</td><td>ΔL* vs background</td><td>0.76</td><td>0.88</td><td>0.81</td><td>0.26</td><td>0.47</td></tr>
<tr><td>Contraction furrow</td><td>ridge filter (strict)</td><td>0.58</td><td>0.99</td><td>0.73</td><td>0.73</td><td>0.60</td></tr>
<tr><td>Lacuna</td><td>difference of Gaussians (strict)</td><td>0.86</td><td>0.60</td><td>0.71</td><td>0.11</td><td>0.04</td></tr>
<tr><td>Dark pigmentation</td><td>ΔE colour difference</td><td>0.69</td><td>0.60</td><td>0.64</td><td>0.28</td><td>0.04</td></tr>
<tr><td>Collarette</td><td>radial gradient</td><td colspan="5">radius within ±0.05 of truth in 88% of synthetic eyes (mean error 0.036)</td></tr>
<tr><td>Fibre type</td><td>structure tensor</td><td colspan="5">radial vs transverse correct in 100% of synthetic eyes</td></tr>
</table>
<p>Settings were chosen on 60 validation irises and reported on 126 test irises. Weak (low-contrast) markings are the main weakness.</p>

<h3>3.5 Speed and storage</h3>
<table>
<tr><th>Measure</th><th>Result</th><th>SRS target</th></tr>
<tr><td>Quick quality check</td><td>~4.5 ms (p95)</td><td>&lt; 2 s</td></tr>
<tr><td>Full two-eye analysis</td><td>0.51 s median, 0.55 s p95 (1 CPU core)</td><td>&lt; 30 s</td></tr>
<tr><td>Models on disk</td><td>3.8 MB network + 2 × 4 KB JSON models (colour, quality)</td><td>–</td></tr>
<tr><td>Code</td><td>~390 KB Python</td><td>–</td></tr>
<tr><td>Libraries</td><td>~330 MB installed (OpenCV, SciPy, ONNX Runtime, NumPy, Pydantic)</td><td>–</td></tr>
<tr><td>Process memory</td><td>~670 MB peak, mostly library overhead</td><td>–</td></tr>
</table>

<h2>4. What was tried and rejected</h2>
<ul>
<li><b>Bigger networks / higher resolution</b> (224, 320 px, MobileNetV3-large): no accuracy gain, 1.5–2.5× slower.</li>
<li><b>INT8 compression</b>: no speed-up on this CPU and a large accuracy loss.</li>
<li><b>Network + classical edge refinement</b>: slightly worse than the network alone.</li>
<li><b>MediaPipe iris landmarks</b>: its radius is a fixed share of the crop, so it only finds the centre.</li>
<li><b>Hough circles, ray-casting</b>: below Daugman everywhere.</li>
<li><b>Colour-cast retake check</b>: misfired on good clinical photos and missed tinted glasses → removed.</li>
<li><b>Periorbital dataset</b>: technically useful (dark irises), but its photos are research-only → not used.</li>
</ul>

<h2>5. Known limitations</h2>
<div class="box warn"><ul>
<li><b>Very dark irises in ordinary light.</b> On the first real phone photos tested (dark brown eyes, glasses, window
reflections), the network mistook the whole iris for the pupil. A plausibility check now rejects such photos with a
clear retake reason instead of producing wrong results, but they cannot yet be analysed. Better capture (bright
indirect side light, no glasses, close-up) or real dark-iris training photos are needed.</li>
<li><b>No real phone photos with masks yet.</b> Segmentation was measured on synthetic colour, real infrared and
visually graded clinical close-ups, none of which is the A-EYE phone domain.</li>
<li><b>Markings are candidates only.</b> They need practitioner review; real-photo accuracy is unknown until
practitioners label A-EYE photos.</li>
<li><b>Licensing.</b> The colour and quality models were fitted on UPOL (academic use) and must be re-fitted on A-EYE
or commercially licensed photos before commercial release; the infrared dataset's origin should be verified.</li>
<li><b>Not a medical device.</b> Outputs describe photographs. Organ or health meanings in the SRS rules are not
scientifically established and are not produced by the model.</li>
</ul></div>

<h2>6. Recommended next steps</h2>
<ol>
<li>Collect 150–300 consented A-EYE phone photos (several phones, lighting conditions, all iris colours, especially dark
irises), with written permission for commercial model development.</li>
<li>Hand-correct iris/pupil outlines on 75–100 of them and fine-tune the 3.8 MB network; re-fit colour and quality models.</li>
<li>Request MOBIUS (signed form, ask about commercial use) for real phone-photo masks and real bad-quality photos.</li>
<li>Have practitioners label real markings, then measure true marking precision/recall and train a small candidate classifier.</li>
</ol>
</body></html>"""


def main():
    from weasyprint import HTML
    OUT.parent.mkdir(parents=True, exist_ok=True)
    HTML(string=html(), base_url=str(ROOT)).write_pdf(str(OUT))
    print("wrote", OUT, f"{OUT.stat().st_size/1e6:.1f} MB")


if __name__ == "__main__":
    main()
