# Datasets used by the benchmark

Raw data is **not committed** (`data/raw/` is git-ignored). Re-download with a free Kaggle
API token (`~/.kaggle/kaggle.json`):

```bash
python scripts/fetch_datasets.py pupil-eye-iris-seg upol synthetic-eyes
python scripts/build_manifest.py          # -> data/manifest.csv (person-level 60/15/25 splits)
```

| Name in manifest | Source | Images used | Ground truth | Licence (as listed) | Role |
|---|---|---|---|---|---|
| `synth_eyes` | Kaggle `allexmendes/synthetic-human-eyes` (50k RGB renders) | 1,573 renders that pass mask QC, cropped 4:3 around the eye with random offset/scale | Pixel masks (pupil/iris/sclera) | CC BY 4.0 | **Visible-light** segmentation train/val/test with exact masks |
| `nir_eye` | Kaggle `itguides/pupil-eye-and-iris-segmentation` | 1,275 grayscale NIR 640x480 | Human-annotated masks: classes 0 bg, 1 eye, **2 pupil, 3 iris** (verified visually) | MIT | Real-photo segmentation train/val/test (NIR domain) |
| `upol` | Kaggle `ouaraskhelilrafik/iris-images` = UPOL Iris DB (Dobeš & Machala) | 384 RGB 576x768 iris close-ups (64 people x 2 eyes x 3) | None (see labels below) | Academic use, cite the authors | **Real visible-light close-ups**: visual segmentation grading, colour labels, marking-injection hosts, end-to-end runs |
| `periorbital` | Zenodo record 13916845 (Nahass et al., *Ophthalmology Science* 2025, "Open-Source Periorbital Segmentation Dataset") | 2,842 two-eye RGB crops from the Chicago Face Database + CelebAMask-HQ → 5,617 single-eye crops (random 4–6 R framing) | Iris (**pupil included, not separately labelled**), sclera, lid, caruncle, brow masks by 5 annotators (inter-grader Dice 0.82) | Annotations **CC BY 4.0**; photos keep their source licences (CFD licence agreement, CelebAMask-HQ non-commercial) → **research only** | Dark irises, diverse skin tones, real lid occlusion: the failure mode seen on real phone photos. Trained with an iris-OR-pupil partial-label loss; evaluated on the iris+pupil union |

**Candidates not yet used:**
* **MOBIUS** (Univ. Ljubljana): 16,717 phone RGB photos, 3,559 hand-corrected iris/pupil/sclera masks, real bad-quality captures. The best fit for A-EYE, but it needs a hand-signed licence form emailed to the authors. Ask them about commercial use.
* **Roboflow Universe** sets listed as CC BY 4.0 (e.g. "YOLO Iris Pupil" ~1,024 images, "FRAMES Iris Seg" 206, "Real Eye Parts 2" 100): they need a free Roboflow account to export (COCO Segmentation) and the site is blocked from this cloud environment. Their image provenance is undocumented, so inspect them before use. To use one: export it, upload the ZIP, and add a loader like `periorbital()`.
* **UTIRIS** (Univ. Tehran): 2048×1360 DSLR RGB iris images, no masks; useful for marking texture.

**Not used:**
* **UBIRIS.v2** (Kaggle `chinmoyslg/ubirisv2`): the images ship inside a **password-protected** archive. The password comes with the authors' research licence. Only the IRISSEG-EP masks were readable, so the dataset could not be used. No attempt was made to bypass the protection. If you obtain the licence, add the images and they slot into `build_manifest.py`.
* MICHE, MOBIUS, CUVIRIS and OpenEDS are not openly downloadable (request forms).

## QC applied to the synthetic renders

The source segmentation sometimes ignores blink/gaze. Renders are kept only if blink ≤ 0.3, squint ≤ 0.5, the sclera is ≥ 20 L* brighter than the masked iris, and the pupil is darker than the iris. A visual spot-check shows about 1 in 10 remaining masks still mismatched. This lowers every method's Dice on this set equally.

## Derived labels committed to this repo

* `data/labels/color_labels.csv`: iris colour appearance (blue / brown / green_mixed / uncertain) for all 384 UPOL images.
  Each of the 128 eyes was labelled once from the contact sheets in `data/labels/sheets/`, and the label applies to all
  3 photos of that eye. One annotator, no adjudication: this is a **pilot** label set. The UPOL camera has a strong warm cast, so blue irises look grey.
* `results/visual*/grades.csv`: pass/fail visual grades of segmentation overlays on UPOL (validation split used for tuning, test split for the final numbers).

## What these datasets can and cannot tell us

* They **can** benchmark capture-quality metrics, iris/pupil localisation (exact masks on synthetic RGB and real NIR; visual grading on real RGB close-ups), normalisation and colour measurement.
* They **cannot** benchmark the six SRS marking categories against practitioner judgement. No public dataset has such labels, so detectors are evaluated by *synthetic injection* into real UPOL iris textures (`bench/markings.py`).
* None of the photos were taken by a phone held 10–20 cm from the eye. **Every threshold, the colour model and ideally the segmenter must be re-calibrated or fine-tuned on consented A-EYE photos.**
