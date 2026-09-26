# T8-3 Architecture Drawings

Automatic metadata extraction for University of Melbourne Archives (UMA) architectural drawing records.

Give the pipeline a folder of scanned drawings and it returns a CSV with the key catalogue fields read from each drawing's title block (project name, drawing number, scale, date, draughtsperson, architect, drawing name, drawing type), plus a confidence score for each field and the source image and Node ID of each drawing. It is designed to reduce manual data entry for archival cataloguing staff.

**Everything runs locally and offline.** No external AI APIs are used. An internet connection is needed only once, during setup.

---

## Contents

1. [How it works](#how-it-works)
2. [Quick start: from a fresh computer to a CSV](#quick-start-from-a-fresh-computer-to-a-csv)
3. [Updating an existing copy](#updating-an-existing-copy)
4. [Running the pipeline](#running-the-pipeline)
5. [Output files](#output-files)
6. [Troubleshooting](#troubleshooting)
7. [Models](#models)
8. [Current status and known limitations](#current-status-and-known-limitations)
9. [For developers](#for-developers)
10. [Project structure](#project-structure)

---

## How it works

```
 drawing (.jpg)
      │
      ▼
 ① Title block detection   Faster R-CNN (fine-tuned)   → finds and crops the title block
      │
      ▼
 ② Text recognition (OCR)  docTR                       → words + their positions
      │
      ▼
 ③ Field classification    LayoutLMv3 (fine-tuned)     → which word belongs to which field
      │
      ▼
 ④ Drawing type            keyword matching            → Plan / Section / Elevation / ...
      │
      ▼
 CSV  (values + per-field confidence + source image + Node ID)
```

| Stage | Tool | Purpose |
|---|---|---|
| 1. Title block detection | Faster R-CNN ResNet-50 FPN (torchvision, fine-tuned) | Locate the title block region on each drawing |
| 2. Text recognition (OCR) | docTR | Extract text and bounding boxes from the title block |
| 3. Field classification | LayoutLMv3 (fine-tuned) | Classify extracted text into metadata fields |
| 4. Drawing type | Keyword matching | Derive `drawing_types` from the extracted `drawing_name` |

---

## Quick start: from a fresh computer to a CSV

Follow these steps in order. Commands are shown for **Mac**, **Windows** and **Linux** (including headless servers).

### Step 0: What you need

| Requirement | Why |
|---|---|
| A GitHub account **added as a collaborator** on this repository | The repository is private (ask the repository owner for access) |
| Python 3.11 or newer | Runs the pipeline |
| Git and the GitHub CLI (`gh`) | Downloads the code and the trained models |
| About 3 GB of free disk space | Python packages (~2 GB) and models (~630 MB) |
| Drawing images (`.jpg`) | The input. Datasets are **not** in the repository |
| Internet, once | Setup downloads packages and models; the first run downloads OCR weights. Afterwards it runs offline |

### Step 1: Install the tools

**Mac** (requires [Homebrew](https://brew.sh)):

```bash
brew install git gh python@3.11
```

**Windows** (PowerShell):

```powershell
winget install --id Git.Git
winget install --id GitHub.cli
winget install --id Python.Python.3.11
```

**Linux** (servers included). Python 3.11 or newer and its `venv` module are required; the system Python is often too old or ships without `venv`:

| Distribution | Command |
|---|---|
| Ubuntu 22.04 (system Python is 3.10) | `sudo apt install git python3.11 python3.11-venv` |
| Ubuntu 24.04 / Debian 12 | `sudo apt install git python3-venv` |
| RHEL / Rocky / Alma 9 | `sudo dnf install git python3.11` |

For `gh`, follow the official Linux instructions at [github.com/cli/cli](https://github.com/cli/cli/blob/trunk/docs/install_linux.md). On a server without `gh`, use a token instead (see [Troubleshooting](#troubleshooting)).

On Windows, **close and reopen the terminal** after installing so the new commands are found. If `winget` is not available, use the installers from [git-scm.com](https://git-scm.com), [cli.github.com](https://cli.github.com) and [python.org](https://www.python.org/downloads/) (tick "Add python.exe to PATH").

Check that everything is installed:

```bash
git --version
gh --version
python3 --version      # Mac: should print 3.11 or newer
py --version           # Windows: should print 3.11 or newer
```

### Step 2: Log in to GitHub

```bash
gh auth login
```

Choose **GitHub.com → HTTPS → Yes (authenticate Git) → Login with a web browser** and follow the prompts. On a server with no browser, `gh` prints a one-time code: open the link it shows on any other device and enter the code there. You only need to do this once per computer. This login is what allows the code and models to be downloaded from the private repository.

### Step 3: Download the project

```bash
gh repo clone tanbowen5566-web/T8-3-Architecture-Drawings
cd T8-3-Architecture-Drawings
```

### Step 4: Run setup

**Mac:**

```bash
python3 setup.py
```

**Windows:**

```bat
py setup.py
```

(If `python3` on Mac reports a version older than 3.11, use `python3.11 setup.py` instead. On Linux, run it the same way as on Mac, with `python3.11 setup.py` on Ubuntu 22.04.)

On **Linux**, setup installs the **CPU-only** build of PyTorch, which is several GB smaller than the default build. The pipeline runs on CPU, so nothing is lost. To install the default (CUDA) build instead, run `python3 setup.py --cuda`.

`setup.py` does everything automatically:

1. Checks the Python version
2. Creates a virtual environment in `venv/`
3. Installs all dependencies from `requirements.txt` (this is the slowest step)
4. Downloads the trained models from the GitHub Release `models-v2` into `output/`, with checksum verification

When it finishes, `output/` should contain `title_block_detector_v2.pth` and a `field_classifier_v2/` folder. If only step 4 failed, the environment is still fine: see [Troubleshooting](#troubleshooting), then run `python download_models.py` after activating the environment.

### Step 5: Activate the environment

Do this every time you open a new terminal to use the pipeline.

| Shell | Command |
|---|---|
| Mac / Linux | `source venv/bin/activate` |
| Windows cmd | `venv\Scripts\activate` |
| Windows PowerShell | `venv\Scripts\Activate.ps1` |

Your prompt now starts with `(venv)`. From here on, `python` works the same on every system.

### Step 6: Add the drawings

Either use one of the team's prepared datasets, or any folder of drawings.

**Option A: a prepared dataset** (gives Node ID traceability). Copy it into `data/` so it looks like this:

```
data/YFA_cabana/
├── metadata.csv          # must have a "Node ID" column
├── 455992/               # one folder per Node ID, containing the .jpg drawings
└── ...
```

Datasets available to the team: `YFA_cabana`, `YFA_state_government`, `Bates_smart_Mccutcheon` (Bates is not yet supported by the detector, see [Current status](#current-status-and-known-limitations)).

**Option B: any folder of drawings.** No preparation needed. Just note the folder's path; no Node ID will be recorded.

### Step 7: Run the pipeline

A quick first run on the first 10 drawings:

```bash
python run_pipeline.py --dataset_name YFA_cabana --run_name demo --limit 10
```

Or on your own folder:

```bash
python run_pipeline.py --input_dir "/path/to/your/drawings" --run_name demo
```

The very first run also downloads docTR's OCR weights (internet needed this once). The pipeline runs on CPU, so large batches take a while; use `--limit` for a first check.

### Step 8: Open the results

The results are in `output/`:

**`output/extracted_metadata_demo.csv`**: one row per drawing, containing the extracted metadata, the confidence score for each field, the source image and the Node ID.

Open it in Excel or any spreadsheet program. See [Output files](#output-files) for details.

---

## Updating an existing copy

If you set the project up before, pull the latest version and refresh dependencies and models:

```bash
git checkout main
git pull
# activate the environment (Step 5), then:
pip install -r requirements.txt
python download_models.py
```

`download_models.py` only downloads models that are missing, so this is safe to run at any time.

---

## Running the pipeline

`run_pipeline.py` runs the whole pipeline (detect and crop title blocks, then OCR and classify fields into a CSV) with a single command. Run it from the project root with the environment activated.

```bash
# Dataset mode: one of the prepared datasets (needs metadata.csv).
# This is the mode that records the Node ID of each drawing.
python run_pipeline.py --dataset_name YFA_cabana --run_name yfa_cabana

# Quick check on the first N images only
python run_pipeline.py --dataset_name YFA_cabana --run_name demo --limit 10

# Folder mode: any folder of raw drawings (e.g. on the Desktop).
# No metadata.csv needed, but no Node ID either.
python run_pipeline.py --input_dir "/path/to/SomeFolder" --run_name some_batch

# No arguments: folder mode over data/YFA_cabana, data/YFA_cabana_test and
# data/YFA_state_government (missing folders are skipped with a warning)
python run_pipeline.py
```

| Option | Meaning |
|---|---|
| `--dataset_name <name>` | Process `data/<name>/` using its `metadata.csv` (dataset mode) |
| `--input_dir <path>` | Process any folder of drawings (folder mode) |
| `--run_name <name>` | Name used for all output files. Defaults to `v2` |
| `--limit <N>` | Only process the first N images |

`--dataset_name` and `--input_dir` cannot be used together.

> **Give every run its own `--run_name`.** Reusing a name overwrites that run's previous output.

---

## Output files

Each run creates, named after `--run_name`:

| File | Contents |
|---|---|
| `output/extracted_metadata_<run_name>.csv` | **Main output**: extracted metadata, per-field confidence, source image and Node ID, one row per drawing |
| `data/titleblock_crops_<run_name>/` | The cropped title block images |
| `output/batch_crop_<run_name>_failed.txt` | Drawings where no title block was confidently detected |

How to read them:

- Each row can be traced back to its drawing through the source image column, and in dataset mode also through the Node ID (left blank in folder mode, where there is no `metadata.csv`).
- Low confidence scores point to values worth checking by hand.
- `drawing_types` (Elevation / Perspective / Plan / Section / Detail) is derived from keywords in the extracted `drawing_name`. If the title contains no keyword, the cell is left blank and should be filled in by hand.
- Drawings listed in `batch_crop_<run_name>_failed.txt` produce **no row** in the CSV.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| Model download fails with **HTTP 404** | You are not logged in, or your account has no access. GitHub reports a private repository's files as "not found" in that case. Run `gh auth login`, confirm you are a collaborator, then `python download_models.py`. Alternatively set a token with read access to the repo: Mac `export GITHUB_TOKEN=<token>`, Windows cmd `set GITHUB_TOKEN=<token>`, PowerShell `$env:GITHUB_TOKEN="<token>"` |
| `command not found: gh` / `git` / `python` | Close and reopen the terminal. On Windows confirm the tool installed; use `py` instead of `python`. On Mac use `python3` before the environment is activated |
| Setup says Python is too old | Install Python 3.11 (Step 1) and run setup with it, e.g. `python3.11 setup.py` (Mac) or `py -3.11 setup.py` (Windows) |
| Linux: `Failed to create virtual environment` / `ensurepip is not available` | The `venv` module is missing. Install it (e.g. `sudo apt install python3.11-venv`, matching your Python version) and re-run setup |
| Linux: `ImportError: libGL.so.1` when the pipeline starts | Minimal server images lack a graphics library that OpenCV loads. Run `sudo apt install libgl1 libglib2.0-0` (Debian/Ubuntu) |
| Linux server: long runs stop when the SSH session closes | Run the pipeline inside `tmux` or `screen`, or prefix the command with `nohup` |
| PowerShell refuses to run `Activate.ps1` | Use Windows cmd instead, or run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once |
| `ModuleNotFoundError` when running the pipeline | The environment is not active. Activate it (Step 5); `(venv)` should appear in the prompt |
| Download blocked by a firewall / want to download by hand | From the repository's **Releases** page, open `models-v2`, put `title_block_detector_v2.pth` in `output/`, and unzip `field_classifier_v2.zip` into `output/` (it contains a `field_classifier_v2/` folder) |
| A drawing is missing from the CSV | Check `output/batch_crop_<run_name>_failed.txt`: no title block was detected for it (see [Known limitations](#known-limitations)) |

Still stuck? Send the full error message to the team.

---

## Models

| Stage | Model | File / source | Size | Distribution |
|---|---|---|---|---|
| Title block detection | Faster R-CNN ResNet-50 FPN, fine-tuned (v2) | `output/title_block_detector_v2.pth` | 158 MB | GitHub Release `models-v2` |
| OCR | docTR `ocr_predictor(pretrained=True)` (default detection + recognition models) | downloaded automatically on first run | — | docTR, internet once |
| Field classification | LayoutLMv3 token classifier, fine-tuned (v2) | `output/field_classifier_v2/` | 478 MB | GitHub Release `models-v2` |

The two fine-tuned models exceed GitHub's 100 MB per-file limit for git, so they are published as **GitHub Release assets** instead of being committed. `python setup.py` downloads them automatically via `download_models.py`, which verifies checksums. To fetch them on their own:

```bash
python download_models.py            # downloads only what is missing
python download_models.py --force    # re-download everything
```

Older versions (`title_block_detector_v0/v1`, `field_classifier_v0/v1`) are superseded by v2 and not distributed. `train_title_block_detector.py` also starts from torchvision's COCO-pretrained Faster R-CNN weights (training only; downloaded automatically).

---

## Current status and known limitations

### Title block detection (`title_block_detector_v2.pth`)

- Trained on 51 annotated images merged from several Label Studio annotation batches, with a train/validation split stratified by source dataset so validation covers both visual styles (`YFA_cabana`, `YFA_state_government`).
- Generalises well across both styles, including scanned and diazo-style drawings that earlier versions failed on.

### Field classification (`field_classifier_v2/`)

- Trained on 84 annotated title block crops (17 held out for validation), with early stopping: the checkpoint with the lowest validation loss is kept, because training beyond about epoch 3 overfits at the current data volume.
- v2 improves every field over v1, for example `scale` recall 70% → 90%, and adds `drawing_name` (held-out recall 88%). Full v2 figures are in the notes in `run_pipeline.py`.
- v1 held-out recall, for reference: `architect` 100%, `date` 96%, `project_name` 91%, `scale` 70%, `drawing_number` 64%, `draughtsperson` 62%.
- Part of the apparent error on `scale`, `drawing_number` and `draughtsperson` comes from a few sparsely annotated validation images (one had only 4 of ~20 fields labelled) rather than model failure.
- Fields with highly variable values (`date`, `draughtsperson`) generalise worse than boilerplate fields (`architect`, `project_name`); more varied annotated examples should help.

### Drawing types

Only keyword matching is used. Visual classifiers were tried and rejected: CLIP discriminated poorly on hand-drafted drawings, and the TU Darmstadt DrawingClassifiers CNN reached about 35% accuracy while covering only 3 of the 5 categories. Drawings whose titles state no type are flagged for manual review instead. The experiments are in `drawing_type_classification/`.

### Known limitations

- **No title block found → no row.** A drawing where the detector's score is below 0.5 produces no CSV row and is only listed in `output/batch_crop_<run_name>_failed.txt`. A dedicated manual-review path is planned but not implemented.
- **Drawings with no OCR text are skipped** as well.
- **OCR misreads** handwritten or stylised characters (e.g. `1/4` read as `V4`). The word is usually still assigned to the correct field, but the value written to the CSV is wrong. Options under consideration: surfacing per-word OCR confidence, or field-specific validation rules.
- **`Bates_smart_Mccutcheon` is not supported yet.** Many of these drawings have no standard title block and are scanned inconsistently; separate handling has not been designed.
- **18 images in `YFA_state_government`** (Node IDs `1264417`–`1264445`) still fail detection with v2, likely due to a shared layout quirk in that batch; not yet investigated.
- **CPU only.** Large batches take a while.

---

## For developers

### Running the two stages separately

Useful if you only need the crops (e.g. for annotation) or want to re-run classification on existing crops without re-detecting:

```bash
# Stage 1 only: detect + crop
python detection/batch_auto_crop_v2.py --input_dir <folder_of_raw_drawings> --output_dir data/<crop_folder>

# Stage 2 only: OCR + field classification -> CSV
python field_classification/export_fields_to_csv.py --image_dir data/<crop_folder> --output_csv output/<name>.csv
```

Both also accept no arguments, falling back to the default datasets and folders used during development.

### Input module

The input module is accessed through `iter_images(dataset_name)` in `src/`. It yields one image record at a time (including `node_id`), so the core pipeline never reads `metadata.csv` or walks the Node ID folders itself. `run_pipeline.py --dataset_name` uses it for Node ID traceability. Output is written by `src/input_output.py::save_output()` as a single CSV containing the metadata fields together with per-field confidence, source image and Node ID.

### Retraining on new annotated data

- **Title block detection:** add the new batch's export folder to `ANNOTATION_BATCHES` in `detection/merge_title_block_annotations.py`, run it to produce a de-duplicated `data/merged_title_block_annotations.json`, then run `detection/train_title_block_detector.py`.
- **Field classification:** convert the new annotation export with `field_classification/convert_annotations_to_layoutlm_format.py`, then run `field_classification/train_field_classifier.py`. All random seeds are fixed (data split, shuffling, classifier head initialisation) so runs are reproducible, and the lowest-validation-loss checkpoint is saved.

Training runs on CPU: MPS has a known numerical instability with torchvision detection models that produces NaN losses. For larger datasets, consider a cloud GPU (e.g. Google Colab).

### Publishing a new model version (maintainers)

1. Create a **new** Release with a new tag (e.g. `models-v3`). Do not replace files inside an existing Release: copies already downloaded would no longer match the checksums.
2. Upload the new `.pth` file and a zip of the field classifier folder (the zip must contain a top-level `field_classifier_vN/` folder).
3. In `download_models.py`, update `RELEASE_TAG` and the `MODELS` file names and checksums (`shasum -a 256 <file>`).
4. Update the model paths in `run_pipeline.py`.
5. Update the tag and file names in this README.

### Contribution rules

- Never commit `venv/`, `data/`, `output/` or any `.pth` file (all are in `.gitignore`).
- Add new dependencies to `requirements.txt` **by hand** with a minimum version (e.g. `foo>=1.2`). Do not use `pip freeze > requirements.txt`: it pins every transitive package to one machine's exact versions and breaks installs on other operating systems.
- Never paste tokens in code, commits or chat.

---

## Project structure

```
.
├── run_pipeline.py                # End-to-end entry point (detect + crop + OCR + classify -> CSV)
├── setup.py                       # Cross-platform setup (venv + dependencies + models)
├── download_models.py             # Downloads trained models from the GitHub Release into output/
├── requirements.txt               # Python dependencies
├── detection/                     # Title block detection: training, annotation merging, inference, visualisation
├── field_classification/          # OCR extraction, annotation prep, training, CSV export
├── drawing_type_classification/   # drawing_types keyword matching + rejected visual-classifier experiments
├── src/                           # Shared input/output module (iter_images, save_output)
├── testing/                       # End-to-end and misc test scripts
├── data/                          # Drawing datasets and annotations (not tracked in git)
├── output/                        # Model weights and generated CSVs (not tracked in git)
└── README.md
```

## Dataset

The architectural drawing datasets were provided by the teaching team. They are stored locally in `data/` and are **not included** in this repository.