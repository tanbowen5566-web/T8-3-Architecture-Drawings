# T8-3 Architecture Drawings

A Python-based project for processing and organising UMA (University of Melbourne Archives) architectural drawing records — automatically detecting title blocks, extracting text via OCR, and classifying the extracted text into structured metadata fields.

## Overview

This tool extracts key metadata (project name, drawing number, scale, drafter, architect, drawing name, and a unique identifier) from the title blocks of historical architectural drawings, outputting the results as CSV with per-field confidence scores. It is designed for archival cataloguing staff to reduce manual data entry.

**Constraint:** the entire pipeline must run fully offline, with no external AI APIs — all models run locally.

## Getting Started

### What you need to do (checklist)

The repository is **private**, and the trained models (~630 MB) are not in git — they are downloaded from the repo's GitHub Release. So each team member needs:

1. **Access to the repository** — you must be added as a collaborator on GitHub (ask the repo owner).
2. **Python 3.11+** and **Git**.
3. **A GitHub login that the model downloader can use** — pick one:
   - **GitHub CLI (recommended):** install [`gh`](https://cli.github.com/) and run `gh auth login` once. `download_models.py` picks the login up automatically.
   - **Or a token:** create a token with read access to the repo (GitHub → Settings → Developer settings → Personal access tokens) and export it before running setup — Mac/Linux: `export GITHUB_TOKEN=<token>`; Windows cmd: `set GITHUB_TOKEN=<token>`.
   - **Or download by hand:** from the repo's Releases page (`model-v2`), download `title_block_detector_v2.pth` into `output/`, and unzip `field_classifier_v2.zip` into `output/` (it contains a `field_classifier_v2/` folder). Setup will report a failed download; that is fine.
4. **The drawing datasets** in `data/` (not in git — copy them from the team's shared copy). See [Data layout](#quick-start-one-command-end-to-end).
5. **Internet once** — for setup (packages + models) and for the first pipeline run (docTR downloads its OCR weights). After that everything runs offline.

Without step 3, the model download fails with HTTP 404 (GitHub reports a private repo's files as "not found" when you are not logged in).

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
4. Downloads the trained models from the GitHub Release into `output/` (see [Models](#models)). If only this step fails, the environment is still fine — fix the login (step 3 above) and re-run `python download_models.py`.

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

`run_pipeline.py` chains the whole pipeline together (detect + crop title blocks, then OCR + classify fields into a CSV) into a single command. Run it from the project root with the virtual environment activated.

**Data layout it expects.** Each dataset lives under `data/` (not in git — copy your own):

```
data/YFA_cabana/
├── metadata.csv          # must have a "Node ID" column
├── 455992/               # one folder per Node ID, containing the .jpg drawings
└── ...
```

Datasets available to the team: `YFA_cabana`, `YFA_state_government`, `Bates_smart_Mccutcheon` (Bates is not supported by the detector yet — see Current Status).

```bash
# Dataset mode: process one of the team's existing datasets (has a metadata.csv).
# This is the mode that gives Node ID traceability in the confidence CSV.
python run_pipeline.py --dataset_name YFA_cabana --run_name yfa_cabana

# Quick demo on just the first N images of a dataset
python run_pipeline.py --dataset_name YFA_cabana --run_name demo --limit 10

# Folder mode: process an arbitrary folder of raw drawings (e.g. one on
# the Desktop). No metadata.csv required, but no Node ID traceability either.
python run_pipeline.py --input_dir "/path/to/SomeFolder" --run_name some_batch

# No arguments: folder mode over data/YFA_cabana, data/YFA_cabana_test and
# data/YFA_state_government (folders that don't exist are skipped with a warning)
python run_pipeline.py
```

`--dataset_name` and `--input_dir` are mutually exclusive. Runs on CPU, so the full YFA datasets take a while; use `--limit 10` for a first check. `--run_name` defaults to `v2`, so **give each run its own name** or it will overwrite the previous run's output.

Each run produces, named after `--run_name`:
- `data/titleblock_crops_<run_name>/` — the cropped title block images
- `output/batch_crop_<run_name>_failed.txt` — images where no title block was confidently detected
- `output/extracted_metadata_<run_name>.csv` — the main output (team's 8-column schema)
- `output/extracted_metadata_<run_name>_confidence.csv` — per-field confidence, plus `source_image`, `node_id` (dataset mode only), `drawing_types_source` and `needs_manual_review`

`drawing_types` (Elevation / Perspective / Plan / Section / Detail) is not predicted by the model: it is derived by keyword matching on the extracted `drawing_name`. If the title contains no keyword the cell is left blank and `needs_manual_review` is `True` in the confidence CSV. The main CSV has no source-image column; rows line up with the confidence CSV by row order.

### Running the two stages separately

The same two stages are also available as standalone scripts, useful if you only need one half of the pipeline (e.g. just want the crops for annotation, or want to re-run classification on crops you already have without re-detecting):

```bash
# Stage 1 only: detect + crop
python detection/batch_auto_crop_v2.py --input_dir <path_to_folder_of_raw_drawings> --output_dir data/<crop_output_folder_name>

# Stage 2 only: OCR + field classification -> CSV
python field_classification/export_fields_to_csv.py --image_dir data/<crop_output_folder_name> --output_csv output/<name>.csv --confidence_csv output/<name>_confidence.csv
```

Both accept no arguments too, in which case they fall back to the original default datasets/folders used during development.

### Known limitations

- Any drawing where the detector finds no title block (score < 0.5) produces **no row** in the CSV; it is only listed in `output/batch_crop_<run_name>_failed.txt`.
- Rows whose OCR finds no text are skipped as well.
- OCR misreads handwritten/stylised characters, so extracted values can be wrong even when the field is classified correctly (see Current Status).

### Retraining on new annotated data

When a new batch of Label Studio annotations comes in:

- **Title block detection**: add the new batch's export folder to `ANNOTATION_BATCHES` in `detection/merge_title_block_annotations.py`, run it to produce a de-duplicated `data/merged_title_block_annotations.json`, then run `detection/train_title_block_detector.py`.
- **Field classification**: convert the new annotation export with `field_classification/convert_annotations_to_layoutlm_format.py`, then run `field_classification/train_field_classifier.py`. This script fixes all random seeds (data split, shuffling, classifier head init) so runs are reproducible, and saves the checkpoint with the lowest validation loss rather than the final epoch, since this model overfits after only a few epochs at current data volumes.

Training currently runs on CPU (MPS has a known numerical instability issue with torchvision's detection models that produces NaN losses). For larger datasets going forward, consider moving training to a cloud GPU (e.g. Google Colab) rather than running long jobs locally.


## Pipeline

| Stage | Tool | Purpose |
|---|---|---|
| 1. Title block detection | Faster R-CNN (torchvision, fine-tuned) | Locate the title block region on each drawing |
| 2. Text recognition (OCR) | docTR | Extract text and bounding boxes from the title block |
| 3. Field classification | LayoutLMv3 (fine-tuned) | Classify extracted text into metadata fields |
| 4. Drawing type | Keyword matching | Derive `drawing_types` from the extracted `drawing_name` |

> Note: the original plan called for RT-DETR for detection. We switched to a fine-tuned Faster R-CNN (torchvision) for the current models — RT-DETR can still be revisited later if detection accuracy needs to improve further.

## Models

Everything the pipeline uses, and where each model comes from:

| Stage | Model | File / source | Size | Distribution |
|---|---|---|---|---|
| Title block detection | Faster R-CNN ResNet-50 FPN, fine-tuned (v2) | `output/title_block_detector_v2.pth` | 158 MB | GitHub Release `model-v2` |
| OCR | docTR `ocr_predictor(pretrained=True)` (default detection + recognition models) | downloaded automatically on first run | — | docTR / internet, once |
| Field classification | LayoutLMv3 token classifier, fine-tuned (v2) | `output/field_classifier_v2/` | 478 MB | GitHub Release `model-v2` |

Older versions (`title_block_detector_v0/v1`, `field_classifier_v0/v1`) are kept locally by the model authors only, are superseded by v2, and are not distributed. `train_title_block_detector.py` also starts from torchvision's COCO-pretrained Faster R-CNN weights (training only; downloaded automatically).

The two fine-tuned models exceed GitHub's 100 MB per-file limit, so they are published as **GitHub Release assets** rather than committed to git. `python setup.py` downloads them into `output/` automatically (via `download_models.py`, with checksum verification). To fetch or re-fetch them on their own:

```bash
python download_models.py            # downloads only what is missing
python download_models.py --force    # re-download everything
```

The repository is private, so it needs a GitHub login: either `gh auth login` (GitHub CLI) or the `GITHUB_TOKEN` environment variable — see the [checklist](#what-you-need-to-do-checklist). Internet access is needed once; the pipeline itself runs fully offline afterwards.

**Publishing a new model version** (maintainers): create a new Release (e.g. tag `model-v3`), upload the new `.pth` file and a zip of the field classifier folder (the zip must contain a top-level `field_classifier_vN/` folder), then update `RELEASE_TAG`, the `MODELS` names + `shasum -a 256` checksums in `download_models.py`, and the model paths in `run_pipeline.py`.

## Current Status

**Title block detection** — `output/title_block_detector_v2.pth`
- Trained on 51 annotated images, merged from multiple Label Studio annotation batches (`detection/merge_title_block_annotations.py` handles de-duplication across batches).
- Uses a stratified train/val split (by source dataset) rather than pure random split, so validation reflects both visual styles in the data (YFA_cabana and YFA_state_government).
- Generalizes reasonably well across both of those styles, including scanned/diazo-style drawings that earlier (smaller-data) versions failed on.
- Known gaps:
  - `Bates_smart_Mccutcheon` is excluded from training/inference for now — these drawings may not have a standard title block region at all, and the images are scanned inconsistently, so this dataset needs separate handling (not yet designed).
  - A cluster of 18 images in `YFA_state_government` (node IDs in the `1264417`–`1264445` range) still fail detection even with the v2 model — likely a shared layout quirk in that batch, not yet investigated.

**Field classification** — `output/field_classifier_v2/` (the v1 figures below are historical; v2 re-includes `drawing_name` and improves every field, e.g. `scale` recall 70% → 90%, see the notes in `run_pipeline.py`)
- Retrained on a larger batch of 84 annotated crops (up from the original 5-image proof of concept), using a stratified train/val split by source dataset (17 held out for validation) and early stopping on validation loss — training past ~epoch 3 overfits noticeably (train loss keeps falling while val loss gets worse), so the checkpoint with the lowest validation loss is saved automatically rather than just the final epoch.
- On the held-out validation split (17 images the model never trained on), per-field recall: `architect` 100%, `date` 96%, `project_name` 91%, `scale` 70%, `drawing_number` 64%, `draughtsperson` 62%. This is a large improvement over the 5-image model, which essentially failed to generalize to any unseen image at all.
- Some of the weaker fields' apparent error rate is inflated by a handful of specific images with incomplete annotation (e.g. one image had only 4 of ~20 expected fields labelled) rather than genuine model failure — worth a spot-check/re-annotation pass on the sparsely-labelled images before drawing firm conclusions about `scale`/`drawing_number`/`draughtsperson` specifically.
- `drawing_name` is included in v2 (held-out recall 88%). `drawing_types` is not predicted by the model: it conflicted with `drawing_name` (often labelled on the same span, while LayoutLMv3 allows one label per word), so it is derived from `drawing_name` by keyword matching instead (see Drawing types below).
- Classification is fairly robust to OCR misreadings (a word can still be routed to the right field even if its text is garbled), but the *value* written to the CSV is only as accurate as the OCR output — this is a separate, still-open problem (see below).
- Fields with highly variable values (`date`, `draughtsperson`) still generalize somewhat worse than fields with consistent boilerplate text (`architect`, `project_name`) — more annotated examples with varied dates/names should help further.

**Drawing types** — only the keyword approach is in the pipeline. Visual classifiers were tried and rejected: CLIP discriminated poorly on these hand-drafted drawings, and the TU Darmstadt DrawingClassifiers CNN reached only ~35% accuracy (and covers just 3 of the 5 categories), so drawings whose titles state no type are flagged for manual review instead. The scripts are in `drawing_type_classification/`.

**OCR** — known issue: handwritten or stylised characters are sometimes misread (e.g. a fraction like `1/4` read as `V4`). This doesn't reliably break field classification (the word still gets routed to the right field based on position/context) but does corrupt the actual extracted value. Not yet addressed — options under consideration: surfacing per-word OCR confidence in the output for manual review, or field-specific validation rules.

**CSV output**
- The main CSV follows the team's 8-column schema via `src/input_output.py::save_output()`.
- Per-field confidence is written to a separate CSV (`*_confidence.csv`) — `save_output()`'s 8-column schema has no room for a source-image or Node ID column, so both live in the confidence CSV instead, joined back to the main CSV by row order.
  - When run via `run_pipeline.py --dataset_name <name>` (see below), the confidence CSV includes a `node_id` column sourced from the dataset's `metadata.csv`, so rows can be traced back to their original Node ID.
  - When run in folder mode (`--input_dir`, no `metadata.csv` available), `node_id` is left blank and only the filename (`source_image`) is available for tracing.
- Images with no detected title block are not yet routed anywhere in particular (see `Bates_smart_Mccutcheon` note above) — a "flag for manual review" path is planned but not implemented.

## Dataset

The architectural drawing dataset was provided by the teaching team for this project. The dataset is stored locally in the `data/` folder and is **not included** in this repository.

## Input Usage
The input module can be accessed through `iter_images(dataset_name)`.
It provides one image record at a time (including `node_id`), so the Core does not need to read `metadata.csv` or search the Node ID folders directly. `run_pipeline.py --dataset_name <name>` uses this directly to get Node ID traceability (see above).

## Notes

- Setup (`python setup.py`) downloads our trained models from the GitHub Release, and the first pipeline run downloads docTR's OCR weights — both need an internet connection **once**. After that, everything runs fully offline. See [Models](#models).
- Do not commit the `venv/`, `data/`, or `output/` folders, or any `.pth` model weight files — all excluded via `.gitignore`. Trained models are distributed through GitHub Releases instead (see [Models](#models)).
- If you add a dependency, add it to `requirements.txt` by hand with a minimum version (e.g. `foo>=1.2`). Avoid `pip freeze > requirements.txt` — it pins every transitive package to your machine's exact versions and breaks installs on other OSes.

## Project Structure

```
.
├── run_pipeline.py       # End-to-end entry point (detect + crop + OCR + classify -> CSV)
├── detection/            # Title block detection: training, merging annotations, inference, visualisation
├── field_classification/ # Field classification: OCR extraction, annotation prep, training, CSV export
├── drawing_type_classification/ # drawing_types keyword matching + rejected visual-classifier experiments
├── testing/              # End-to-end and misc test scripts
├── src/                  # Shared input/output module (iter_images, save_output)
├── data/                 # Drawing dataset and annotations (not tracked in git)
├── output/               # Trained model weights and generated CSVs (not tracked in git)
├── requirements.txt      # Python dependencies
├── setup.py              # Cross-platform environment setup script (venv + dependencies + models)
├── download_models.py    # Downloads trained models from the GitHub Release into output/
└── README.md
```