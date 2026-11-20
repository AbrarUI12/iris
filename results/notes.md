### Why `auto` and not the programmatic pixel-metric winner

The rule above picks `onnx|seg_small_160` because it wins on the datasets that have masks
(visible-light synthetic renders and real NIR). The real-photo check changes the answer:
on **real visible-light iris close-ups** (UPOL; the closest public match to iridology/A-EYE
macro photos) the learned model fails on 30/30 held-out images (the limbus is badly
underestimated; it never saw real high-resolution iris texture filling the frame), while
the classical Daugman operator passes 28/30.

The shipped default is therefore **`auto`** (`aeye_iris/pipeline.yaml`). It always runs Daugman (~40–60 ms). If the
iris radius is ≥ 0.40 of the short image side (an iris-filling close-up) it keeps that result.
Otherwise it runs the 3.8 MB MobileNetV3-small/160 model (~8–12 ms). The 0.40 threshold came
from the *validation* splits only: close-ups are ≥ 0.43 (5th percentile) and whole-eye framing is ≤ 0.38 (95th percentile).

| | `auto` (default) | learned small_160 | Daugman (best classical) |
|---|---|---|---|
| Real NIR iris Dice (held-out subjects) | 0.913 | **0.917** | 0.571 |
| Visible-light synthetic iris Dice | 0.779 | **0.795** | 0.609 |
| Real close-ups, visual pass (30 held-out) | **28/30** | 0/30 | **28/30** |
| Latency per eye, 1 thread | ~30–55 ms | ~8–12 ms | ~16–60 ms |
| Model size | 3.8 MB | 3.8 MB | 0 |

### What was tried and rejected (and why)

* **Bigger / higher-resolution networks** (MobileNetV3-large 224, small 224/320): no accuracy gain over
  small_160 on any test set, 1.5–2.5x slower → rejected.
* **INT8 static quantisation** (ONNX Runtime QDQ): no speed-up on this x86 CPU and large accuracy
  loss (small_160: NIR Dice 0.92 → 0.64 even with percentile calibration and a float head). The large model
  keeps more accuracy (0.89) but is slower than FP32 small_160. Would need quantisation-aware training → rejected.
* **Model + classical edge refinement (`hybrid`, band)**: slightly worse than the model alone everywhere → rejected.
* **Model pupil + classical limbus search (`hybrid`, search)**: 27/30 on real close-ups but worse
  on periocular framing than `auto` → rejected in favour of `auto`.
* **MediaPipe iris landmarks**: its iris radius is a fixed fraction (~9%) of the crop regardless of the true
  size (measured), so it is only a centre proposer here; with a classical radius search it reaches
  0.43 (NIR) / 0.54 (synthetic) iris Dice → rejected.
* **Hough circles**: worst classical method (0.33–0.46 iris Dice, and the slowest at up to 100+ ms with CLAHE) → rejected. **Ray-cast + RANSAC ellipse**:
  below Daugman on every set.
* **Preprocessing**: CLAHE is the single most useful option for all classical methods (+0.04–0.08 Dice);
  red vs gray vs V channel differ by < 0.01; specular-reflection inpainting matters on NIR (+0.08 Dice
  with CLAHE: strong corneal glints) and is neutral on RGB, so it stays on.
* **Colour**: hand-set CIELAB rules reach only 0.52 macro-F1 under a warm camera cast; a 5-feature
  logistic model (≈ 20 numbers, numpy inference) reaches 0.88 → learned, but must be re-fitted per camera / on A-EYE photos.
* **Quality gate**: a logistic model over all metrics (AUC 0.91) beats per-metric thresholds (AUC 0.78);
  JPEG artefacts are barely detected by any metric.

### Data reality check

* UBIRIS.v2 could not be used: the Kaggle mirror ships the images inside a password-protected archive
  (the password is issued by the authors with the research licence); only its masks were readable.
* About 1 in 10 Synthetic-Human-Eyes masks still disagree with the image after automatic QC (eyelid/gaze
  mismatch in the source data), which caps every method's measured Dice on that set (same for all methods).
* No public dataset labels the six SRS markings; their numbers come from synthetic injection only.

### Self-training on real close-ups (UPOL train pseudo-labels)

`scripts/pseudo_label_upol.py` labelled the UPOL **train** split with Daugman, keeping 130 of 198 images that passed strict automatic QC (pupil/iris ratio, concentricity, edge strength, close-up framing). Val and test images were never used. small_160 was then fine-tuned for 10 epochs with these images oversampled ×4 (`models/seg_small_160_st.onnx`).

| | self-trained small_160 | default `auto` | acceptance rule (set before the run) |
|---|---|---|---|
| Real close-ups, 30 held-out (visual) | **29/30** | 28/30 | ≥ 28/30 ✔ |
| Synthetic RGB iris Dice | **0.789** | 0.779 | drop ≤ 0.01 ✔ |
| Real NIR iris Dice | 0.895 | 0.913 | drop ≤ 0.01 ✘ (−0.018) |
| Latency per eye | ~12–23 ms (one model) | ~30–55 ms (Daugman + model) | – |

It fails the NIR criterion, so **the default stays `auto`**. The self-trained model is the leading candidate for the visible-light A-EYE domain: enable it with `segmentation: {name: onnx, params: {model: seg_small_160_st.onnx}}`. Caveat: pseudo-labels come from Daugman, so the model can inherit its systematic errors. Promote it only after an independent check on hand-annotated A-EYE phone photos.
