# A-EYE iris pipeline (iris only)

Lightweight, LLM-free image pipeline for the iris part of the A-EYE SRS:

```
photo -> quality precheck -> iris/pupil segmentation -> quality assess
      -> rubber-sheet normalisation -> colour category -> 12 sectors x 3 rings
      -> marking CANDIDATES (6 SRS types) -> ScanObservation JSON (+ overlay PNG)
```

Outputs are **descriptive image measurements** with explicit `unassessable` /
`not_detected_in_assessable_image` / `present_candidate` states. Nothing here infers
organ function or health; zone labels are practitioner configuration (`ZoneMap.labels`).

**Which combination is best, and why:** see [`results/BENCHMARK.md`](results/BENCHMARK.md). In short:

| Stage | Shipped default | Why (measured) |
|---|---|---|
| Segmentation | `auto`: Daugman (CLAHE, red channel, glint inpainting) for iris-filling close-ups, MobileNetV3-small/160 ONNX (3.8 MB) otherwise | Real close-ups: 28/30 pass (learned alone 0/30); whole-eye framing: Dice 0.91 NIR / 0.78 RGB (Daugman 0.57 / 0.61) |
| Quality gate | Explainable rules (+ logistic score option) | Logistic AUC 0.91 vs rules 0.78; precheck ~5 ms |
| Colour | 5-feature CIELAB logistic model, `uncertain` band | macro-F1 0.88 (hand rules 0.52); re-fit per camera |
| Markings | ridge furrows, ΔL white, DoG lacuna, ΔE pigment, gradient collarette, structure-tensor fibres | Synthetic-injection recall 0.67–1.0 at full strength; **not practitioner-validated** |
| Speed | Two-eye scan ~0.5 s on 1 CPU core | SRS budget 30 s |

## Quick start

Running on your own laptop / GPU: see [`docs/SETUP_LAPTOP.md`](docs/SETUP_LAPTOP.md).


```bash
pip install -e ".[bench,train,dev]"
python -m aeye_iris.pipeline --left left.jpg --right right.jpg --out runs/scan1
pytest -q
```

## Layout

| Path | What |
|---|---|
| `aeye_iris/segment/classical.py` | Daugman integro-differential, Hough, ray-cast + RANSAC ellipse (training-free) |
| `aeye_iris/segment/learned.py` | ONNX MobileNetV3 U-Net (`onnx`), model + classical edge refinement (`hybrid`), MediaPipe iris landmarks (`mediapipe`) |
| `aeye_iris/quality.py` | explainable accept / retake / review gate, per-feature assessability |
| `aeye_iris/normalize.py`, `zones.py` | rubber sheet, 12x3 clock-face zones (documented orientation convention) |
| `aeye_iris/color.py` | blue / brown / green_mixed / uncertain (rules or tiny learned model, numpy-only inference) |
| `aeye_iris/markings/` | dark pigmentation, white marking, lacuna, contraction furrow, collarette, fibre orientation |
| `aeye_iris/pipeline.py` + `pipeline.yaml` | config-driven end-to-end run (default = benchmark winner) |
| `train/` | LightIrisNet-style segmenter training on CPU or CUDA, ONNX export, INT8 static quantisation |
| `synth/` | procedural eyes, marking injection, capture degradations (benchmark ground truth) |
| `bench/` | segmentation grid, quality gate, colour, markings, end-to-end, report |

## Reproduce the benchmark

```bash
python scripts/fetch_datasets.py pupil-eye-iris-seg upol synthetic-eyes && python scripts/build_manifest.py
for cfg in small:160 small:224 small:320 large:224; do b=${cfg%:*}; s=${cfg#*:}
  python -m train.train_seg --backbone $b --size $s --datasets synth_eyes nir_eye --out models/seg_${b}_${s}
  python -m train.quantize_int8 models/seg_${b}_${s}.onnx; done
python -m bench.seg --config bench/configs/seg_all.yaml --datasets synth_eyes nir_eye
python -m bench.visual_sheet --split test --n 30 --configs "auto|model=seg_small_160.onnx"   # real close-ups, graded by eye
python -m bench.quality --segmenter auto --segkw '{"model":"seg_small_160.onnx"}'
python -m bench.color && python -m bench.markings && python -m bench.e2e
python -m bench.report
```
