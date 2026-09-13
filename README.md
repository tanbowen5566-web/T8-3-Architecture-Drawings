# T8-3 Architecture Drawings

A Python-based project for processing and organising UMA (University of Melbourne Archives) architectural drawing records — automatically detecting title blocks, extracting text via OCR, and classifying the extracted text into structured metadata fields.

## Overview

This tool extracts key metadata (project name, drawing number, scale, drafter, architect, drawing name, and a unique identifier) from the title blocks of historical architectural drawings, outputting the results as CSV with per-field confidence scores. It is designed for archival cataloguing staff to reduce manual data entry.

**Constraint:** the entire pipeline must run fully offline, with no external AI APIs — all models run locally.

## Pipeline

| Stage | Tool | Purpose |
|---|---|---|
| 1. Title block detection | Faster R-CNN (torchvision, fine-tuned) | Locate the title block region on each drawing |
| 2. Text recognition (OCR) | docTR | Extract text and bounding boxes from the title block |
| 3. Field classification | LayoutLMv3 (fine-tuned) | Classify extracted text into metadata fields |

> Note: the original plan called for RT-DETR for detection. We switched to a fine-tuned Faster R-CNN (torchvision) for the current models — RT-DETR can still be revisited later if detection accuracy needs to improve further.

## Current Status

**Title block detection** — `output/title_block_detector_v2.pth`
- Trained on 51 annotated images, merged from multiple Label Studio annotation batches (`detection/merge_title_block_annotations.py` handles de-duplication across batches).
- Uses a stratified train/val split (by source dataset) rather than pure random split, so validation reflects both visual styles in the data (YFA_cabana and YFA_state_government).
- Generalizes reasonably well across both of those styles, including scanned/diazo-style drawings that earlier (smaller-data) versions failed on.
- Known gaps:
  - `Bates_smart_Mccutcheon` is excluded from training/inference for now — these drawings may not have a standard title block region at all, and the images are scanned inconsistently, so this dataset needs separate handling (not yet designed).
  - A cluster of 18 images in `YFA_state_government` (node IDs in the `1264417`–`1264445` range) still fail detection even with the v2 model — likely a shared layout quirk in that batch, not yet investigated.

**Field classification** — `output/field_classifier_v1/`
- Retrained on a larger batch of 84 annotated crops (up from the original 5-image proof of concept), using a stratified train/val split by source dataset (17 held out for validation) and early stopping on validation loss — training past ~epoch 3 overfits noticeably (train loss keeps falling while val loss gets worse), so the checkpoint with the lowest validation loss is saved automatically rather than just the final epoch.
- On the held-out validation split (17 images the model never trained on), per-field recall: `architect` 100%, `date` 96%, `project_name` 91%, `scale` 70%, `drawing_number` 64%, `draughtsperson` 62%. This is a large improvement over the 5-image model, which essentially failed to generalize to any unseen image at all.
- Some of the weaker fields' apparent error rate is inflated by a handful of specific images with incomplete annotation (e.g. one image had only 4 of ~20 expected fields labelled) rather than genuine model failure — worth a spot-check/re-annotation pass on the sparsely-labelled images before drawing firm conclusions about `scale`/`drawing_number`/`draughtsperson` specifically.
- `drawing_name` and `drawing_types` are excluded from the current model. In annotation, these two fields were often labelled on the same span of text, which conflicts with LayoutLMv3's one-label-per-word setup — needs a team decision on how the two fields should be defined/scoped before re-including them.
- Classification is fairly robust to OCR misreadings (a word can still be routed to the right field even if its text is garbled), but the *value* written to the CSV is only as accurate as the OCR output — this is a separate, still-open problem (see below).
- Fields with highly variable values (`date`, `draughtsperson`) still generalize somewhat worse than fields with consistent boilerplate text (`architect`, `project_name`) — more annotated examples with varied dates/names should help further.

**OCR** — known issue: handwritten or stylised characters are sometimes misread (e.g. a fraction like `1/4` read as `V4`). This doesn't reliably break field classification (the word still gets routed to the right field based on position/context) but does corrupt the actual extracted value. Not yet addressed — options under consideration: surfacing per-word OCR confidence in the output for manual review, or field-specific validation rules.

**CSV output**
- The main CSV follows the team's 8-column schema via `src/input_output.py::save_output()`.
- Per-field confidence is written to a separate CSV (`*_confidence.csv`) — `save_output()`'s 8-column schema has no room for a source-image or Node ID column, so both live in the confidence CSV instead, joined back to the main CSV by row order.
  - When run via `run_pipeline.py --dataset_name <name>` (see below), the confidence CSV includes a `node_id` column sourced from the dataset's `metadata.csv`, so rows can be traced back to their original Node ID.
  - When run in folder mode (`--input_dir`, no `metadata.csv` available), `node_id` is left blank and only the filename (`source_image`) is available for tracing.
- Images with no detected title block are not yet routed anywhere in particular (see `Bates_smart_Mccutcheon` note above) — a "flag for manual review" path is planned but not implemented.

## Dataset

The architectural drawing dataset was provided by the teaching team for this project. The dataset is stored locally in the `data/` folder and is **not included** in this repository.

## Getting Started

### Prerequisites

- Python 3.11 or higher
- Git

### Setup

```bash
git clone <repo-url>
cd T8-3-Architecture-Drawings
python setup.py
```

`setup.py` automatically:
1. Checks your Python version
2. Creates a virtual environment (`venv/`)
3. Installs all dependencies from `requirements.txt`

This works on both **Mac/Linux** and **Windows** — no need to run separate scripts.

### Activate the environment

**Mac/Linux:**
```bash
source venv/bin/activate
```

**Windows:**
```bat
venv\Scripts\activate
```

## Running the pipeline

### Quick start: one command, end to end

`run_pipeline.py` chains the whole pipeline together (detect + crop title blocks, then OCR + classify fields into a CSV) into a single command. This supersedes the earlier `run_ocr_pipeline.py`, which was an explicitly-labelled rule-based stopgap (keyword matching + position heuristics) used before the LayoutLMv3 field classifier existed — that script has since been removed from the repo (see git history if you need to refer back to it). See [Known differences from `run_ocr_pipeline.py`](#known-differences-from-run_ocr_pipelinepy) below for the coverage it had that the current pipeline doesn't yet match.

```bash
# Dataset mode: process one of the team's existing datasets (has a metadata.csv).
# This is the mode that gives Node ID traceability in the confidence CSV.
python run_pipeline.py --dataset_name YFA_cabana --run_name yfa_cabana

# Quick demo on just the first N images of a dataset
python run_pipeline.py --dataset_name YFA_cabana --run_name demo --limit 10

# Folder mode: process an arbitrary folder of raw drawings (e.g. one on
# the Desktop). No metadata.csv required, but no Node ID traceability either.
python run_pipeline.py --input_dir "/Users/dazai/Desktop/SomeFolder" --run_name some_batch
```

Each run produces, named after `--run_name`:
- `data/titleblock_crops_<run_name>/` — the cropped title block images
- `output/batch_crop_<run_name>_failed.txt` — images where no title block was confidently detected
- `output/extracted_metadata_<run_name>.csv` — the main output (team's 8-column schema)
- `output/extracted_metadata_<run_name>_confidence.csv` — per-field confidence, plus `source_image` and (in dataset mode) `node_id`

### Running the two stages separately

The same two stages are also available as standalone scripts, useful if you only need one half of the pipeline (e.g. just want the crops for annotation, or want to re-run classification on crops you already have without re-detecting):

```bash
# Stage 1 only: detect + crop
python detection/batch_auto_crop_v2.py --input_dir <path_to_folder_of_raw_drawings> --output_dir data/<crop_output_folder_name>

# Stage 2 only: OCR + field classification -> CSV
python field_classification/export_fields_to_csv.py --image_dir data/<crop_output_folder_name> --output_csv output/<name>.csv --confidence_csv output/<name>_confidence.csv
```

Both accept no arguments too, in which case they fall back to the original default datasets/folders used during development.

### Known differences from `run_ocr_pipeline.py`

`run_pipeline.py` is the intended long-term replacement (per `run_ocr_pipeline.py`'s own docstring), but two gaps remain before it's a full drop-in:
- `drawing_name` is not currently output at all (the field classifier doesn't predict it yet — see Current Status above). `run_ocr_pipeline.py` attempted it via a fragile position-based heuristic, so this is a loss in that one field's coverage, not a regression in reliability.
- `run_ocr_pipeline.py` used the OCR model directly on the *full, uncropped* drawing rather than a title-block crop — this pipeline always crops first via the detection model, which is the intended design but means any drawing the detector fails on (see Current Status) currently produces no row at all, rather than a low-quality attempt at one.

### Retraining on new annotated data

When a new batch of Label Studio annotations comes in:

- **Title block detection**: add the new batch's export folder to `ANNOTATION_BATCHES` in `detection/merge_title_block_annotations.py`, run it to produce a de-duplicated `data/merged_title_block_annotations.json`, then run `detection/train_title_block_detector.py`.
- **Field classification**: convert the new annotation export with `field_classification/convert_annotations_to_layoutlm_format.py`, then run `field_classification/train_field_classifier.py`. This script fixes all random seeds (data split, shuffling, classifier head init) so runs are reproducible, and saves the checkpoint with the lowest validation loss rather than the final epoch, since this model overfits after only a few epochs at current data volumes.

Training currently runs on CPU (MPS has a known numerical instability issue with torchvision's detection models that produces NaN losses). For larger datasets going forward, consider moving training to a cloud GPU (e.g. Google Colab) rather than running long jobs locally.

## Input Usage
The input module can be accessed through `iter_images(dataset_name)`.
It provides one image record at a time (including `node_id`), so the Core does not need to read `metadata.csv` or search the Node ID folders directly. `run_pipeline.py --dataset_name <name>` uses this directly to get Node ID traceability (see above).

## Notes

- First run will download model weights (docTR, LayoutLMv3, torchvision Faster R-CNN) — this requires an internet connection **once**. After that, the models are cached locally and can run fully offline.
- Do not commit the `venv/`, `data/`, or `output/` folders, or any `.pth` model weight files — all excluded via `.gitignore`. Trained models currently only exist locally on the machine that trained them; share them directly (e.g. via cloud storage) if a teammate needs to use them without retraining.
- If you're contributing new dependencies, regenerate `requirements.txt` with:
  ```bash
  pip freeze > requirements.txt
  ```

## Project Structure

```
.
├── run_pipeline.py       # End-to-end entry point (detect + crop + OCR + classify -> CSV)
├── detection/            # Title block detection: training, merging annotations, inference, visualisation
├── field_classification/ # Field classification: OCR extraction, annotation prep, training, CSV export
├── testing/              # End-to-end and misc test scripts
├── src/                  # Shared input/output module (iter_images, save_output)
├── data/                 # Drawing dataset and annotations (not tracked in git)
├── output/               # Trained model weights and generated CSVs (not tracked in git)
├── requirements.txt      # Python dependencies
├── setup.py              # Cross-platform environment setup script
└── README.md
```