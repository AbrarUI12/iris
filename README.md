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

**Which combination is best, and why:** see [`results/BENCHMARK.md`](results/BENCHMARK.md).

## Quick start

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
| `train/` | LightIrisNet-style segmenter training on CPU, ONNX export, INT8 static quantisation |
| `synth/` | procedural eyes, marking injection, capture degradations (benchmark ground truth) |
| `bench/` | segmentation grid, quality gate, colour, markings, end-to-end, report |

## Reproduce the benchmark

```bash
python scripts/fetch_datasets.py && python scripts/build_manifest.py
for cfg in small:160 small:224 small:320 large:224; do b=${cfg%:*}; s=${cfg#*:}
  python -m train.train_seg --backbone $b --size $s --out models/seg_${b}_${s}
  python -m train.quantize_int8 models/seg_${b}_${s}.onnx; done
python -m bench.seg --config bench/configs/seg_classical.yaml
python -m bench.seg --config bench/configs/seg_learned.yaml --out results/seg_learned
python -m bench.quality && python -m bench.color && python -m bench.markings && python -m bench.e2e
python -m bench.report
```
