# T8-3 Architecture Drawings

Automatic metadata extraction for University of Melbourne Archives (UMA) architectural drawing records.

Give the pipeline a folder of scanned drawings and it returns a CSV with the key catalogue fields read from each drawing's title block, or from the whole sheet when a drawing has no title block (project name, drawing number, scale, date, draughtsperson, architect, drawing name, drawing type), plus a confidence score for each field and the source image and Node ID of each drawing. It is designed to reduce manual data entry for archival cataloguing staff.

**Everything runs locally and offline.** No external AI APIs are used. An internet connection is needed only once, during setup.

---

## Contents

1. [Quick start: from a fresh computer to a CSV](#quick-start-from-a-fresh-computer-to-a-csv)
2. [How it works](#how-it-works)
3. [Updating an existing copy](#updating-an-existing-copy)
4. [Commands and options](#commands-and-options)
5. [Output files](#output-files)
6. [Troubleshooting](#troubleshooting)
7. [Models](#models)
8. [Current status and known limitations](#current-status-and-known-limitations)
9. [For developers](#for-developers)
10. [Project structure](#project-structure)

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
4. Downloads the trained models from the GitHub Release `models-v2` into `models/`, with checksum verification

When it finishes, `models/` should contain `title_block_detector_v2.pth` and a `field_classifier_v2/` folder. If only step 4 failed, the environment is still fine: see [Troubleshooting](#troubleshooting), then run `python download_models.py` after activating the environment.

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

Datasets available to the team: `YFA_cabana`, `YFA_state_government`, `Bates_smart_Mccutcheon` (most Bates drawings have no title block and go through the No-TB branch, see [How it works](#how-it-works)).

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

The results are in `results/`:

**`results/extracted_metadata_demo.csv`**: one row per drawing, containing the extracted metadata, the confidence score for each field, the source image and the Node ID.

Open it in Excel or any spreadsheet program. See [Output files](#output-files) for details.

---

## How it works

Every drawing first goes through title block detection. Drawings **with** a title block are read from the cropped title block (TB branch); drawings **without** one are read from the whole sheet (No-TB branch). Both branches write to the same CSV.

```
 drawing (.jpg)
        │
        ▼
 ① Title block detection   Faster R-CNN (fine-tuned)
        │
        ├──────────────────────────────────────────┐
        │ title block found (score ≥ 0.5)          │ no title block (score < 0.5)
        ▼ TB branch                                ▼ No-TB branch (whole sheet)
 ② crop the title block                     ② preprocess: rescale, flatten paper, contrast
 ③ OCR (docTR) on the crop                  ③ OCR (docTR) on overlapping tiles
 ④ field classification                     ④ rules: text blocks → block roles → fields
    LayoutLMv3 (fine-tuned)                    (title, view title, scale, drawing no., stamp …)
        │                                          │
        └─────────────────────┬────────────────────┘
                              ▼
 ⑤ Drawing type            keyword matching      → Plan / Section / Elevation / ...
                              │
                              ▼
 CSV  (values + per-field confidence + source image + Node ID + branch)
```

| Stage | Branch | Tool | Purpose |
|---|---|---|---|
| 1. Title block detection | both | Faster R-CNN ResNet-50 FPN (torchvision, fine-tuned) | Locate the title block and decide which branch the drawing goes to |
| 2. Crop | TB | — | Cut out the detected title block |
| 2. Preprocessing | No-TB | OpenCV | Bring every sheet to a similar size, remove the yellowed paper background, boost faint lettering |
| 3. Text recognition (OCR) | both | docTR | Extract words and their positions (No-TB: on overlapping tiles so small lettering is not lost) |
| 4. Field classification | TB | LayoutLMv3 (fine-tuned) | Classify each word into a metadata field |
| 4. Rule-based extraction | No-TB | `src/no_tb/` | Group words into text blocks, give each block one role (title, view title, scale, drawing number, stamp, contract note, archive mark …), then read each field only from blocks of the matching role |
| 5. Drawing type | both | Keyword matching | Derive `drawing_types` from the extracted `drawing_name` |

**Why the No-TB branch uses rules.** The LayoutLMv3 model was trained on title block crops only; on whole sheets it labels almost every word as "no field". It is kept as a baseline for comparison (`--no_tb_method layoutlm`, see [Commands and options](#commands-and-options)).

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

## Commands and options

Reference for everything `run_pipeline.py` can do. For a first run, follow the [Quick start](#quick-start-from-a-fresh-computer-to-a-csv) instead. Run all commands from the project root with the environment activated.

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

# No-TB branch: save a debug overlay per drawing (to check / tune the rules)
python run_pipeline.py --dataset_name Bates_smart_Mccutcheon --run_name bsm --limit 10 --no_tb_debug

# Skip drawings without a title block (behaviour before the No-TB branch existed)
python run_pipeline.py --dataset_name YFA_cabana --run_name tb_only --no_tb skip
```

**General options**

| Option | Meaning |
|---|---|
| `--dataset_name <name>` | Process `data/<name>/` using its `metadata.csv` (dataset mode) |
| `--input_dir <path>` | Process any folder of drawings (folder mode) |
| `--run_name <name>` | Name used for this run's output: `results/extracted_metadata_<name>.csv` and `intermediate/<name>/`. Defaults to `v2` |
| `--limit <N>` | Only process the first N images |

`--dataset_name` and `--input_dir` cannot be used together.

**No-TB branch options** (drawings where no title block is detected)

| Option | Meaning |
|---|---|
| `--no_tb skip` | Do not process drawings without a title block; they are only listed in `intermediate/<run_name>/no_tb_images.txt`. Default: they are processed |
| `--no_tb_method rules` / `layoutlm` | `rules` (default): rule-based extractor. `layoutlm`: baseline, the TB field classifier applied to the whole sheet (for comparison only) |
| `--no_tb_debug` | For every No-TB drawing, write a JSON file (each text block, its role and why) and an overlay image (blocks coloured by role) to `intermediate/<run_name>/no_tb_debug/` |
| `--no_tb_target_long <px>` | `rules` only. Long side the sheet is resized to before OCR. Default `4096`; lower (e.g. `3072`) is faster but may miss small lettering |
| `--no_tb_ocr full` / `tiled`, `--no_tb_upscale <x>` | `layoutlm` baseline only: OCR on the whole sheet at once or on tiles, and a resize factor before OCR |

> **Give every run its own `--run_name`.** Reusing a name overwrites that run's `results/` CSV and its `intermediate/<run_name>/` folder.

---

## Output files

Each run creates, named after `--run_name`:

| File | Contents |
|---|---|
| `results/extracted_metadata_<run_name>.csv` | **Main output**: one row per drawing (TB and No-TB), see the columns below |
| `intermediate/<run_name>/routing.csv` | Which branch each drawing went to, with the detector score and the crop path |
| `intermediate/<run_name>/no_tb_images.txt` | Drawings where no title block was detected (processed by the No-TB branch unless `--no_tb skip`) |
| `intermediate/<run_name>/titleblock_crops/` | The cropped title block images (TB branch) |
| `intermediate/<run_name>/no_tb_debug/` | Only with `--no_tb_debug`: per-drawing JSON + overlay image |

Everything under `intermediate/` can be deleted once a run is checked; only `results/` is the deliverable.

**Columns of the main CSV**

| Column | Meaning |
|---|---|
| `source_image` | File name of the drawing |
| `node_id` | Node ID from `metadata.csv` (dataset mode only; blank in folder mode) |
| `<field>`, `<field>_confidence` | Each field (`project_name`, `drawing_name`, `architect`, `draughtsperson`, `drawing_number`, `date`, `scale`, `drawing_types`) followed by its confidence from 0 to 1. Both are blank when the field was not found |
| `branch` | `tb` = read from the title block, `no_tb` = read from the whole sheet |
| `tb_score` | Title block detector score (below 0.5 → `no_tb`) |
| `drawing_types_source` | `keyword` = a type keyword was found in `drawing_name`; `unmatched` = none found |
| `needs_manual_review` | `TRUE` if `drawing_types` is `unmatched`, or if the row comes from the No-TB branch |

How to read them:

- Each row can be traced back to its drawing through `source_image`, and in dataset mode also through `node_id`.
- Low confidence scores point to values worth checking by hand.
- `drawing_types` (Elevation / Perspective / Plan / Section / Detail) is derived from keywords in the extracted `drawing_name`. Its confidence of 1.000 means the keyword was found, not that `drawing_name` itself was read correctly: check `drawing_name_confidence` for that. If there is no keyword, the cell is left blank and should be filled in by hand.
- Rows with `branch = no_tb` come from a first version of the rule-based extractor and are always flagged for review (see [Known limitations](#known-limitations)).

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
| Download blocked by a firewall / want to download by hand | From the repository's **Releases** page, open `models-v2`, put `title_block_detector_v2.pth` in `models/`, and unzip `field_classifier_v2.zip` into `models/` (it contains a `field_classifier_v2/` folder) |
| A drawing is missing from the CSV | Either OCR found no text on it (the terminal shows `OCR found no text, skipping`), or the run used `--no_tb skip` and the drawing is listed in `intermediate/<run_name>/no_tb_images.txt`. `intermediate/<run_name>/routing.csv` shows which branch each drawing went to |
| A No-TB row has wrong or empty fields | Re-run that drawing with `--no_tb_debug` and open its overlay in `intermediate/<run_name>/no_tb_debug/`: blocks without a box were not read by OCR; a box with the wrong role means a rule needs adjusting (`src/no_tb/field_specs.py`) |
| `Warning: using model from old location 'output/...'` | Models are still in the old `output/` folder. Move them: `mkdir -p models && mv output/title_block_detector_v*.pth output/field_classifier_v* models/` |

Still stuck? Send the full error message to the team.

---

## Models

| Stage | Model | File / source | Size | Distribution |
|---|---|---|---|---|
| Title block detection | Faster R-CNN ResNet-50 FPN, fine-tuned (v2) | `models/title_block_detector_v2.pth` | 158 MB | GitHub Release `models-v2` |
| OCR | docTR `ocr_predictor(pretrained=True)` (default detection + recognition models) | downloaded automatically on first run | — | docTR, internet once |
| Field classification | LayoutLMv3 token classifier, fine-tuned (v2) | `models/field_classifier_v2/` | 478 MB | GitHub Release `models-v2` |

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

### No-TB branch (`src/no_tb/`)

- First version of a rule-based extractor for drawings without a title block. The rules were derived from a small number of `Bates_smart_Mccutcheon` sheets (contract sections / elevations and site plans) and have **not yet been validated against ground truth**, so every No-TB row has `needs_manual_review = TRUE`.
- Typical source of each field on these sheets: `project_name` = the title lettering (often 2–3 lines), `drawing_name` = all view titles (e.g. `SECTION A.B; SECTION C.D`), `drawing_number` = `Nº3` / `Sheet 16`, `scale` = text such as `EIGHT FEET TO ONE INCH`, `architect` = an architect's rubber stamp, `date` = the handwritten contract note.
- Tuning is done in `src/no_tb/field_specs.py` (keywords, patterns, thresholds), using the `--no_tb_debug` overlays.

### Drawing types

Only keyword matching is used. Visual classifiers were tried and rejected: CLIP discriminated poorly on hand-drafted drawings, and the TU Darmstadt DrawingClassifiers CNN reached about 35% accuracy while covering only 3 of the 5 categories. Drawings whose titles state no type are flagged for manual review instead. The experiments are in `drawing_type_classification/`.

### Known limitations

- **No-TB results are unvalidated.** Drawings without a title block (detector score below 0.5) are processed by the rule-based No-TB branch, whose output is always flagged for review.
- **Handwriting is largely unreadable.** On No-TB sheets the date is usually only in a handwritten contract note, so `date` is often empty; signatures are those of contractors and witnesses and are never used as `architect`. `architect` and `draughtsperson` are often simply not on these sheets.
- **Drawings with no OCR text are skipped** as well.
- **OCR misreads** handwritten or stylised characters (e.g. `1/4` read as `V4`). The word is usually still assigned to the correct field, but the value written to the CSV is wrong. Options under consideration: surfacing per-word OCR confidence, or field-specific validation rules.
- **Rotated text** (vertical street names, diagonal stamps) is not read by the current OCR setting, so an architect's stamp printed at an angle may be missed.
- **18 images in `YFA_state_government`** (Node IDs `1264417`–`1264445`) still fail detection with v2, likely due to a shared layout quirk in that batch; not yet investigated. They are now processed by the No-TB branch instead of being dropped.
- **CPU only.** Large batches take a while; No-TB sheets are the slowest (OCR runs on 15–20 tiles per sheet, see `--no_tb_target_long`).

---

## For developers

### Running the two stages separately

Useful if you only need the crops (e.g. for annotation) or want to re-run classification on existing crops without re-detecting:

```bash
# Stage 1 only: detect + crop
python detection/batch_auto_crop_v2.py --input_dir <folder_of_raw_drawings> --output_dir data/<crop_folder>

# Stage 2 only: OCR + field classification -> CSV
python field_classification/export_fields_to_csv.py --image_dir data/<crop_folder> --output_csv results/<name>.csv
```

Both also accept no arguments, falling back to the default datasets and folders used during development.

### Input module

The input module is accessed through `iter_images(dataset_name)` in `src/`. It yields one image record at a time (including `node_id`), so the core pipeline never reads `metadata.csv` or walks the Node ID folders itself. `run_pipeline.py --dataset_name` uses it for Node ID traceability. Output is written by `src/input_output.py::save_output()` as a single CSV containing the metadata fields together with per-field confidence, source image and Node ID.

### Tests

Unit tests for the No-TB rules run without any model:

```bash
pip install pytest
python -m pytest tests/test_no_tb_rules.py -q
```

### Tuning the No-TB branch

1. Run a small batch with `--no_tb_debug` (see [Commands and options](#commands-and-options)).
2. Open the overlays in `intermediate/<run_name>/no_tb_debug/`. Box colours: red = title, orange = view title, green = scale, purple = drawing number, pink = stamp, blue = contract note, light blue = other label, grey = archive mark / dimension / unreadable.
3. For a block with the wrong role, the `why` entry in the matching JSON file says which rule assigned it.
4. Adjust keywords, patterns or thresholds in `src/no_tb/field_specs.py` (logic lives in `roles.py` and `extractor.py`), add a case to `tests/test_no_tb_rules.py`, and re-run the tests.

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

- Never commit `venv/`, `data/`, `models/`, `intermediate/`, `results/` or any `.pth` file (all are in `.gitignore`).
- Add new dependencies to `requirements.txt` **by hand** with a minimum version (e.g. `foo>=1.2`). Do not use `pip freeze > requirements.txt`: it pins every transitive package to one machine's exact versions and breaks installs on other operating systems.
- Never paste tokens in code, commits or chat.

---

## Project structure

```
.
├── run_pipeline.py                # End-to-end entry point (detect → TB / No-TB branch → CSV)
├── setup.py                       # Cross-platform setup (venv + dependencies + models)
├── download_models.py             # Downloads trained models from the GitHub Release into models/
├── requirements.txt               # Python dependencies
├── detection/                     # Title block detection: training, annotation merging, inference, visualisation
├── field_classification/          # OCR extraction, annotation prep, training, CSV export
├── drawing_type_classification/   # drawing_types keyword matching + rejected visual-classifier experiments
├── src/                           # Shared input/output module (iter_images, save_output)
│   └── no_tb/                     # No-TB branch: preprocessing, tiled OCR, block roles, field rules
├── tests/                         # Unit tests (pytest), e.g. test_no_tb_rules.py
├── testing/                       # End-to-end and misc test scripts
├── data/                          # Drawing datasets and annotations (not tracked in git)
├── models/                        # Trained model weights (not tracked in git)
├── intermediate/                  # Files produced during a run, one folder per run_name (not tracked in git)
├── results/                       # Final output CSVs (not tracked in git)
└── README.md
```

## Dataset

The architectural drawing datasets were provided by the teaching team. They are stored locally in `data/` and are **not included** in this repository.