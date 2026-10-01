# T8-3 Architecture Drawings

Automatic metadata extraction for University of Melbourne Archives (UMA) architectural drawing records.

Give the pipeline a folder of scanned drawings and it returns a CSV with the key catalogue fields read from each drawing's title block, or from the whole sheet when a drawing has no title block (project name, drawing number, scale, date, draughtsperson, architect, drawing name, drawing type), plus a confidence score for each field and the source image and Node ID of each drawing. It is designed to reduce manual data entry for archival cataloguing staff.

**Everything runs locally and offline.** No external AI APIs are used. An internet connection is needed only once, during setup.

---

## Contents

1. [Quick start: from a fresh computer to a CSV](#quick-start-from-a-fresh-computer-to-a-csv)
   (Step 7 covers [getting updates](#step-7-getting-updates))
2. [How it works](#how-it-works)
3. [Commands and options](#commands-and-options)
4. [Output files](#output-files)
5. [Troubleshooting](#troubleshooting)
6. [Models](#models)
7. [Current status and known limitations](#current-status-and-known-limitations)
8. [For developers](#for-developers)
9. [Project structure](#project-structure)

---

## Quick start: from a fresh computer to a CSV

Seven steps, in order. Each step has instructions for **Mac**, **Windows** and **Linux**; follow only the ones for your computer. Commands go in a terminal: **Terminal** on Mac (Applications → Utilities), **PowerShell** on Windows (Start menu → type "PowerShell").

You will need about 3 GB of free disk space and an internet connection during setup. After that, the pipeline runs offline.

### Step 1: Get the project

There are two ways to get the project. Use the one that matches how you received it.

#### Option A: you received a zip file

1. **Unzip it.** Mac: double-click the zip. Windows: right-click the zip → **Extract All** → **Extract** (do not open files from inside the zip without extracting first).
2. **Move the unzipped folder** somewhere permanent, e.g. your Documents folder.
3. **Open a terminal in that folder:**
   - **Mac:** open Terminal, type `cd ` (with a space after it), drag the project folder from Finder into the Terminal window, and press Enter.
   - **Windows:** open the project folder in File Explorer, click the address bar at the top, type `powershell` and press Enter.
   - **Linux:** `cd /path/to/T8-3-Architecture-Drawings`

If the zip already contains a `models/` folder, no GitHub account is needed: go straight to [Step 2](#step-2-run-setup). If it does not, setup will download the models from GitHub, which requires access to the repository: do steps 1 and 2 of Option B (install `gh`, then log in) before running setup.

#### Option B: download from GitHub

This needs a GitHub account that has been **added as a collaborator** on the repository (ask the repository owner), because the repository is private.

1. **Install Git and the GitHub CLI** (one time only):

   | System | Command |
   |---|---|
   | Mac | `brew install git gh` (requires [Homebrew](https://brew.sh), see [Step 2](#mac)) |
   | Windows | `winget install --id Git.Git` and then `winget install --id GitHub.cli`. Close and reopen PowerShell afterwards. No `winget`? Use the installers from [git-scm.com](https://git-scm.com/downloads) and [cli.github.com](https://cli.github.com) |
   | Linux | `sudo apt install git` (or `sudo dnf install git`), then `gh` from the [official Linux instructions](https://github.com/cli/cli/blob/trunk/docs/install_linux.md) |

2. **Log in to GitHub** (one time per computer):

   ```bash
   gh auth login
   ```

   Choose **GitHub.com → HTTPS → Yes → Login with a web browser**, and follow the prompts. On a server with no browser, `gh` shows a one-time code: open the link it gives on any other device and enter the code there.

3. **Download (clone) the project and enter its folder:**

   ```bash
   gh repo clone tanbowen5566-web/T8-3-Architecture-Drawings
   cd T8-3-Architecture-Drawings
   ```

   This creates a folder called `T8-3-Architecture-Drawings` inside whichever folder your terminal was in (usually your home folder).

### Step 2: Run setup

Setup needs **Python 3.11 or newer**. Install it first if you don't have it, then run setup **from inside the project folder** (the terminal from Step 1).

#### Mac

1. **Check your Python version:** `python3 --version`. If it prints 3.11 or higher, skip to 3.
2. **Install Python 3.11**, either way:
   - **Easiest:** download the macOS installer from [python.org](https://www.python.org/downloads/macos/) and run it. Then open **Applications → Python 3.11** and double-click **Install Certificates.command** (this prevents download errors later).
   - **With Homebrew:** install [Homebrew](https://brew.sh) by pasting this into Terminal, and run the "Next steps" commands it prints at the end:

     ```bash
     /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
     ```

     Then: `brew install python@3.11`
3. **Run setup.** If you installed Python 3.11 in step 2:

   ```bash
   python3.11 setup.py
   ```

   If `python3 --version` already showed 3.11 or higher: `python3 setup.py`

#### Windows

1. **Check your Python version:** `py --version`. If it prints 3.11 or higher, skip to 3.
2. **Install Python 3.11**, either way:
   - `winget install --id Python.Python.3.11`, then close and reopen PowerShell (and go back to the project folder as in Step 1).
   - Or download the installer from [python.org](https://www.python.org/downloads/windows/) (64-bit). On the first screen, tick **"Add python.exe to PATH"**.
3. **Run setup:**

   ```bat
   py setup.py
   ```

   (`py` automatically uses the newest Python installed.)

#### Linux

1. **Install Python 3.11 and its `venv` module** (the system Python is often too old or lacks `venv`):

   | Distribution | Command |
   |---|---|
   | Ubuntu 22.04 (system Python is 3.10) | `sudo apt install python3.11 python3.11-venv` |
   | Ubuntu 24.04 / Debian 12 | `sudo apt install python3-venv` |
   | RHEL / Rocky / Alma 9 | `sudo dnf install python3.11` |

2. **Run setup:** `python3.11 setup.py` (on Ubuntu 24.04 / Debian 12: `python3 setup.py`)

On Linux, setup installs the **CPU-only** build of PyTorch, which is several GB smaller than the default. The pipeline runs on CPU, so nothing is lost. To install the default (CUDA) build instead, add `--cuda`.

#### What setup does

Setup takes several minutes; installing dependencies is the slowest part. It:

1. Checks the Python version
2. Creates a virtual environment in `venv/` (a private copy of Python for this project)
3. Installs the dependencies from `requirements.txt`
4. Downloads the trained models from the GitHub Release `models-v2` into `models/` (skipped for models that are already there)

### Step 3: Check that setup succeeded

At the end, setup prints a **summary** of its five steps. **Success** looks like this:

```
============================================================
 Setup summary
============================================================
 ✓ 1. Check Python version         Python 3.11
 ✓ 2. Find requirements.txt        found
 ✓ 3. Create virtual environment   created ./venv
 ✓ 4. Install dependencies         installed
 ✓ 5. Download trained models      models/ ready

==> Setup complete!
```

`models/` now contains `title_block_detector_v2.pth` and a `field_classifier_v2/` folder. Go to [Step 4](#step-4-activate-the-environment).

**If a step failed**, the summary marks it with ✗ and, below the list, says what went wrong (**Problem**) and what to do (**Fix**), for example:

```
 ✓ 1. Check Python version         Python 3.11
 ✓ 2. Find requirements.txt        found
 ✓ 3. Create virtual environment   created ./venv
 ✗ 4. Install dependencies         FAILED
 – 5. Download trained models      not run

✗ Step 4 failed: Install dependencies
  Problem: pip failed while installing the packages in requirements.txt (exit code 1).
  Fix:     Scroll up to the first line starting with 'ERROR:' just above this summary; it names the cause.
           ...
```

Follow the **Fix**, then run setup again ([Step 2](#step-2-run-setup)). Steps that already succeeded are reused, so it continues where it stopped. If only step 5 failed, the environment is usable but the pipeline cannot run until the models are in place.

The table below covers every failed step, plus errors that appear before setup can even start.

| Where it fails | Message | What to do |
|---|---|---|
| Before setup starts (Mac/Linux) | `command not found: python3` / `command not found: python3.11` | Python 3.11 is not installed. Install it as in [Step 2](#mac). If you just installed it, close and reopen Terminal |
| Before setup starts (Windows) | `Python was not found; run without arguments to install from the Microsoft Store…` or `'py' is not recognized…` | Python is not installed or not on PATH. Install it as in [Step 2](#windows), with **"Add python.exe to PATH"** ticked, then close and reopen PowerShell |
| Before setup starts | `can't open file '…setup.py': [Errno 2] No such file or directory` | The terminal is not in the project folder. Go back into it as in [Step 1](#step-1-get-the-project) (`cd` into the folder that contains `setup.py`) |
| Step 1: Check Python version | `Current Python version is 3.9, but this project requires Python 3.11 or higher.` (any version below 3.11) | Setup was started with an older Python. Install 3.11 as in [Step 2](#step-2-run-setup), then run setup with the version written explicitly: `python3.11 setup.py` (Mac/Linux) or `py -3.11 setup.py` (Windows) |
| Step 2: Find requirements.txt | `requirements.txt is missing from the project folder` | The project folder is incomplete. Get the project again ([Step 1](#step-1-get-the-project)) |
| Step 3: Create virtual environment | `'… -m venv venv' failed` | Linux: install the `venv` package shown under Fix (e.g. `sudo apt install python3.11-venv`). Mac/Windows: reinstall Python from [python.org](https://www.python.org/downloads/) |
| Step 3: Create virtual environment | `The existing 'venv' folder is broken and could not be deleted` | Close any terminal or editor using the project, delete the `venv` folder by hand, run setup again. (A broken or outdated `venv` folder is normally rebuilt automatically) |
| Step 4: Install dependencies | `pip failed while …` | Scroll up to the first line starting with `ERROR:`. Connection errors: check the internet connection and run setup again. `No space left on device`: free about 3 GB. `Could not find a version that satisfies the requirement torch…`: you are using 32-bit Python or a version PyTorch does not support yet; install 64-bit Python 3.11 and run setup with `py -3.11` / `python3.11` |
| Step 5: Download trained models | `GitHub answered 'not found' (HTTP 404)` | GitHub does not know who you are, or your account has no access to the private repository. Do [Option B steps 1–2](#option-b-download-from-github) (install `gh`, `gh auth login`), make sure you have been added as a collaborator, then run setup again. Or download the models by hand (next rows) |
| Step 5: Download trained models | `GitHub rejected the login (HTTP 401/403)` | The saved login or token has expired. Run `gh auth login` again, then run setup again |
| Step 5: Download trained models | `Python could not verify GitHub's SSL certificate` (Mac) | Python from python.org needs its certificates installed once: open **Applications → Python 3.11** and double-click **Install Certificates.command**, then run setup again |
| Step 5: Download trained models | `A downloaded model file is corrupted (checksum mismatch)` | Activate the environment ([Step 4](#step-4-activate-the-environment)), then run `python download_models.py --force` |
| Step 5: Download trained models | `Could not reach GitHub` (no internet, firewall) | Check the connection and run setup again, or download by hand: on the repository's **Releases** page open `models-v2`, put `title_block_detector_v2.pth` in `models/`, and unzip `field_classifier_v2.zip` into `models/` (it contains a `field_classifier_v2/` folder) |
| Any step | `! Setup was interrupted during step …` | Setup was stopped (e.g. Ctrl+C). Run it again; finished steps are reused |

Anything else: send the full text in the terminal to the team.

### Step 4: Activate the environment

Do this **every time** you open a new terminal to use the pipeline, from inside the project folder.

| System | Command |
|---|---|
| Mac / Linux | `source venv/bin/activate` |
| Windows PowerShell | `venv\Scripts\Activate.ps1` |
| Windows Command Prompt (cmd) | `venv\Scripts\activate` |

It worked if the line in your terminal now starts with `(venv)`. From here on, type `python` on every system (not `python3` or `py`).

If PowerShell says **`Activate.ps1 cannot be loaded because running scripts is disabled on this system`**, run this once and then activate again:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

### Step 5: Add the drawings

The drawings are not part of the project; you add them yourself. Choose one option.

#### Option A: a dataset folder with `metadata.csv` (records Node IDs)

Use this when your drawings come with a `metadata.csv`, so every row of the result can be traced back to its Node ID.

1. **Check the folder layout.** It must look like this:

   ```
   YFA_cabana/                 ← dataset folder (any name)
   ├── metadata.csv            ← must have a column named exactly "Node ID"
   ├── 455992/                 ← one folder per Node ID, named exactly as in metadata.csv
   │   └── 455992_001.jpg      ← the drawings (.jpg)
   ├── 455993/
   └── ...
   ```

2. **Drag the whole dataset folder into the `data` folder** inside the project folder (create a folder named `data` there if it doesn't exist). The result is `data/YFA_cabana/metadata.csv`, `data/YFA_cabana/455992/…`.

Datasets used by the team: `YFA_cabana`, `YFA_state_government`, `Bates_smart_Mccutcheon` (most Bates drawings have no title block and go through the No-TB branch, see [How it works](#how-it-works)).

#### Option B: any folder of drawings (no Node IDs)

Use this for a plain folder of `.jpg` drawings anywhere on your computer (e.g. on the Desktop). No `metadata.csv` and no moving needed, but the result will have no Node ID.

You only need to copy the folder's **path** (its full location), which goes into the command in Step 6:

- **Mac:** in Finder, right-click the folder, hold the **Option** key, and choose **Copy "…" as Pathname**.
- **Windows:** in File Explorer, right-click the folder → **Copy as path** (on Windows 10, hold **Shift** while right-clicking).

### Step 6: Run the pipeline

Make sure `(venv)` is shown (Step 4). Copy the command for your option and change only the parts in **CAPITALS**.

#### Option A: dataset in `data/`

```bash
python run_pipeline.py --dataset_name YFA_cabana --run_name MY_FIRST_RUN --limit 10
```

| Part | Change it to |
|---|---|
| `YFA_cabana` | The name of **your** dataset folder inside `data/`, spelled exactly as the folder |
| `MY_FIRST_RUN` | Any new name for this run, using letters, numbers and `_` only (no spaces). Use a **different name for every run**: reusing a name overwrites that run's results |
| `--limit 10` | Processes only the first 10 drawings, for a quick check. **Delete `--limit 10`** to process the whole dataset |

#### Option B: any folder

```bash
python run_pipeline.py --input_dir "/PATH/TO/YOUR/FOLDER" --run_name MY_FIRST_RUN --limit 10
```

| Part | Change it to |
|---|---|
| `"/PATH/TO/YOUR/FOLDER"` | The path you copied in Step 5. The quotes must stay, because paths often contain spaces. **Mac:** paste between the quotes, e.g. `"/Users/anna/Desktop/Drawings 1965"`. **Windows:** the copied path already has quotes, so replace the quotes and everything between them, e.g. `"C:\Users\anna\Desktop\Drawings 1965"` |
| `MY_FIRST_RUN`, `--limit 10` | As in Option A |

#### While it runs

The terminal shows progress for each drawing. The very first run also downloads the OCR model weights once (internet needed). Everything runs on the computer's processor, so a full dataset can take a long time; drawings without a title block are the slowest.

#### Your results

When it finishes, the result is in the `results` folder inside the project folder:

**`results/extracted_metadata_MY_FIRST_RUN.csv`**: one row per drawing, with the extracted fields, a confidence score (0 to 1) for each field, the source image, the Node ID (Option A only), and a `needs_manual_review` column marking rows to check by hand.

Open it in Excel or any spreadsheet program. See [Output files](#output-files) for every column.

### Step 7: Getting updates

When the team releases a new version:

#### If you got the project as a zip (Option A in Step 1)

1. Unzip the new version into a **new** folder.
2. Move your `data` folder from the old project folder into the new one.
3. Run setup again in the new folder ([Step 2](#step-2-run-setup)), then continue from [Step 4](#step-4-activate-the-environment).

Results from earlier runs stay in the old folder's `results/`. Once you have copied what you need, you can delete the old folder.

#### If you cloned from GitHub (Option B in Step 1)

From inside the project folder, with the environment activated ([Step 4](#step-4-activate-the-environment)):

```bash
git checkout main
git pull
pip install -r requirements.txt
python download_models.py
```

`git pull` gets the new code, `pip install` adds any new dependencies, and `download_models.py` fetches any new models (it only downloads what is missing, so it is safe to run every time). Your `data/` and `results/` folders are not touched.

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
| Setup fails (see also [Step 3](#step-3-check-that-setup-succeeded)): Python is too old | Install Python 3.11 ([Step 2](#step-2-run-setup)) and run setup with it, e.g. `python3.11 setup.py` (Mac) or `py -3.11 setup.py` (Windows) |
| Linux: `Failed to create virtual environment` / `ensurepip is not available` | The `venv` module is missing. Install it (e.g. `sudo apt install python3.11-venv`, matching your Python version) and re-run setup |
| Linux: `ImportError: libGL.so.1` when the pipeline starts | Minimal server images lack a graphics library that OpenCV loads. Run `sudo apt install libgl1 libglib2.0-0` (Debian/Ubuntu) |
| Linux server: long runs stop when the SSH session closes | Run the pipeline inside `tmux` or `screen`, or prefix the command with `nohup` |
| PowerShell refuses to run `Activate.ps1` | Use Windows cmd instead, or run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once |
| `ModuleNotFoundError` when running the pipeline | The environment is not active. Activate it ([Step 4](#step-4-activate-the-environment)); `(venv)` should appear in the prompt |
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