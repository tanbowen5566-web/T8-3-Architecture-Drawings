# T8-3 Architecture Drawings

Automatic metadata extraction for University of Melbourne Archives (UMA) architectural drawing records.

Give the pipeline a folder of scanned drawings and it returns a CSV with the key catalogue fields read from each drawing's title block, or from the whole sheet when a drawing has no title block (project name, drawing number, scale, date, draughtsperson, architect, drawing name, drawing type), plus a confidence score for each field and the source image and Node ID of each drawing. It is designed to reduce manual data entry for archival cataloguing staff.

**Everything runs locally and offline.** No external AI APIs are used. An internet connection is needed only once, during setup.

---

## Contents

1. [Quick start: from a fresh computer to a CSV](#quick-start-from-a-fresh-computer-to-a-csv)
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

This guide starts from the project folder you received. Setup needs about 3 GB of free disk space and an internet connection; after that, the pipeline runs offline.

**Where to type the commands:** open the project folder in [VS Code](https://code.visualstudio.com) (**File → Open Folder…** and choose the project folder), then open its terminal with **Terminal → New Terminal**. This terminal already starts in the project folder, so every command below can be copied into it as it is. Each command is in its own box: hover over the box and click the copy icon on its right.

### Step 1: Run setup

#### Mac

1. Check your Python version:

   ```bash
   python3.11 --version
   ```

   If it shows 3.11 or higher, go straight to step 3.

2. Install Python 3.11, either way:
   - **Easiest:** run the macOS installer from [python.org](https://www.python.org/downloads/macos/). Then open **Applications → Python 3.11** and double-click **Install Certificates.command**.
   - **With [Homebrew](https://brew.sh):**

     ```bash
     brew install python@3.11
     ```

   Then close the terminal and open a new one (**Terminal → New Terminal**).

3. Run setup.:

   ```bash
   python3.11 setup.py
   ```

#### Windows

1. Check your Python version:

   ```powershell
   py --version
   ```

   If it shows 3.11 or higher, go straight to step 3.

2. Install Python 3.11, either with this command or with the 64-bit installer from [python.org](https://www.python.org/downloads/windows/) (tick **"Add python.exe to PATH"**):

   ```powershell
   winget install --id Python.Python.3.11
   ```

   Then close the terminal and open a new one (**Terminal → New Terminal**).

3. Run setup:

   ```powershell
   py setup.py
   ```

#### Linux

1. Install Python 3.11 and its `venv` module, with the command for your distribution:

   Ubuntu 22.04:

   ```bash
   sudo apt install python3.11 python3.11-venv
   ```

   Ubuntu 24.04 / Debian 12:

   ```bash
   sudo apt install python3-venv
   ```

   RHEL / Rocky / Alma 9:

   ```bash
   sudo dnf install python3.11
   ```

2. Run setup:

   ```bash
   python3.11 setup.py
   ```

   On Ubuntu 24.04 / Debian 12:

   ```bash
   python3 setup.py
   ```

   Linux gets the smaller CPU-only build of PyTorch. To install the default (CUDA) build instead, for a machine with an NVIDIA GPU:

   ```bash
   python3.11 setup.py --cuda
   ```

Setup takes several minutes. It checks Python, creates a private Python environment in `venv/`, installs the dependencies and puts the trained models in `models/`.

### Step 2: Check that setup succeeded

Setup ends with a summary. **Success** looks like this:

```text
 ✓ 1. Check Python version         Python 3.11
 ✓ 2. Find requirements.txt        found
 ✓ 3. Create virtual environment   created ./venv
 ✓ 4. Install dependencies         installed
 ✓ 5. Download trained models      models/ ready

==> Setup complete!
```

**If a step failed**, it is marked ✗, and below the list the summary shows the **Problem** and the **Fix**. Follow the Fix and run setup again; finished steps are reused. The most common cases:

- **`command not found: python3`** (Mac), or **`Python was not found`** / **`'py' is not recognized`** (Windows): Python is not installed. Install it as in [Step 1](#step-1-run-setup), then open a new terminal.
- **`can't open file '…setup.py'`**: VS Code has a different folder open. Use **File → Open Folder…** to open the project folder itself (the one that contains `setup.py`), then open a new terminal.
- **✗ 1. Check Python version**: setup was started with an older Python. Run it with the version named. Mac / Linux:

  ```bash
  python3.11 setup.py
  ```

  Windows:

  ```powershell
  py -3.11 setup.py
  ```

- **✗ 3. Create virtual environment** (Linux): the `venv` module is missing. Install it, then run setup again:

  ```bash
  sudo apt install python3.11-venv
  ```

- **✗ 4. Install dependencies**: usually the internet connection dropped. Check it and run setup again. Otherwise, the first line starting with `ERROR:` above the summary names the cause.
- **✗ 5. Download trained models**: everything else is already set up; only the two model files are missing. Get them from GitHub instead:
  1. Open the [Releases page](https://github.com/tanbowen5566-web/T8-3-Architecture-Drawings/releases) and find the release **`models-v2`**.
  2. Under **Assets**, download **`title_block_detector_v2.pth`** and **`field_classifier_v2.zip`**.
  3. In the project folder, open the **`models`** folder. If there is none, create a folder named exactly `models` (in VS Code: right-click in the file list on the left → **New Folder…**).
  4. Move **`title_block_detector_v2.pth`** into `models`.
  5. Unzip **`field_classifier_v2.zip`** (Mac: double-click it; Windows: right-click → **Extract All**) and move the resulting **`field_classifier_v2`** folder into `models`.

  If a file or folder with the same name is already in `models`, replace it. The result must look like this:

```text
  models/
  ├── title_block_detector_v2.pth
  └── field_classifier_v2/
      ├── config.json
      └── ...
```

  On Windows, **Extract All** sometimes creates a folder inside a folder (`field_classifier_v2/field_classifier_v2/`). If so, move the inner folder into `models` so that `models/field_classifier_v2/config.json` exists.

  There is no need to run setup again: continue with [Step 3](#step-3-activate-the-environment).

### Step 3: Activate the environment

Do this **every time** you open a new terminal to use the pipeline. Skip it if the line in the terminal already starts with `(venv)`.

Mac / Linux:

```bash
source venv/bin/activate
```

Windows (PowerShell, the default terminal in VS Code):

```powershell
venv\Scripts\Activate.ps1
```

Windows (Command Prompt):

```bat
venv\Scripts\activate
```

It worked if the line in your terminal now starts with `(venv)`. From here on, type `python` on every system (not `python3` or `py`).

If PowerShell says **`Activate.ps1 cannot be loaded because running scripts is disabled on this system`**, run this once and then activate again:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

### Step 4: Add the drawings

#### 1. Check the folder layout

The pipeline only accepts drawings in a **dataset folder** with exactly this layout:

```text
YFA_cabana/                      ← dataset folder (any name)
├── metadata.csv                 ← directly in the dataset folder; must have a column named exactly "Node ID"
├── 455992/                      ← one folder per Node ID, named exactly as in metadata.csv
│   └── 1238196_….jpg            ← the drawing image(s) of that Node ID (.jpg, any file name)
├── 455993/
│   └── ….jpg
└── ...
```

Check three things: `metadata.csv` sits directly inside the dataset folder (not in a subfolder), every drawing is inside a Node ID folder, and every Node ID folder name appears in the `Node ID` column of `metadata.csv`.

#### 2. Choose where the folder goes

Both options give the same result. They differ only in where the dataset folder is kept, which changes the command in [Step 5](#step-5-run-the-pipeline).

- **Option A: inside the project.** Drag the whole dataset folder into the `data` folder inside the project folder (you can drag it straight into VS Code's file list on the left; create a folder named `data` if it doesn't exist). `data` can hold several datasets side by side:

  ```text
  data/
  ├── Bates_smart_Mccutcheon/
  ├── YFA_cabana/
  │   ├── metadata.csv
  │   ├── 455992/
  │   └── ...
  └── YFA_state_government/
  ```

  In Step 5 you then refer to the dataset by its **folder name**.

- **Option B: anywhere else** (e.g. on the Desktop or an external drive). Leave the dataset folder where it is (same layout as above) and copy the **path of the dataset folder itself** (e.g. of `YFA_cabana`, not of a Node ID folder inside it); in Step 5 you then refer to it by this path.
  - **Mac:** in Finder, right-click the folder, hold the **Option** key, and choose **Copy "…" as Pathname**.
  - **Windows:** in File Explorer, right-click the folder → **Copy as path** (on Windows 10, hold **Shift** while right-clicking).

### Step 5: Run the pipeline

Make sure the terminal line starts with `(venv)` ([Step 3](#step-3-activate-the-environment)). Copy the command for your option, paste it into the terminal, and change only the parts in **CAPITALS** before pressing Enter.
> [!CAUTION]
> **After pressing Enter, the terminal may show nothing for a minute or two.**

#### Option A: dataset in `data/`

```bash
python run_pipeline.py --dataset_name YFA_cabana --run_name MY_FIRST_RUN --limit 10
```

| Part | Change it to |
|---|---|
| `YFA_cabana` | The name of **your** dataset folder inside `data/`, spelled exactly as the folder |
| `MY_FIRST_RUN` | Any new name for this run, using letters, numbers and `_` only (no spaces). Use a **different name for every run**: reusing a name overwrites that run's results |
| `--limit 10` | Processes only the first 10 drawings, for a quick check. **Delete `--limit 10`** to process the whole dataset |

#### Option B: dataset anywhere else

```bash
python run_pipeline.py --input_dir "/PATH/TO/YOUR/FOLDER" --run_name MY_FIRST_RUN --limit 10
```

| Part | Change it to |
|---|---|
| `"/PATH/TO/YOUR/FOLDER"` | The path you copied in Step 4. The quotes must stay, because paths often contain spaces. **Mac:** paste between the quotes, e.g. `"/Users/anna/Desktop/YFA_cabana"`. **Windows:** the copied path already has quotes, so replace the quotes and everything between them, e.g. `"C:\Users\anna\Desktop\YFA_cabana"` |
| `MY_FIRST_RUN`, `--limit 10` | As in Option A |

The pipeline is loading its models, and the very first run also downloads the OCR model weights once (internet needed), which can take a few minutes more. It is not frozen: don't close the terminal or press Enter again. Once loading is done, the terminal shows progress for each drawing.
>
> Everything runs on the computer's processor, so a full dataset can take a long time; drawings without a title block are the slowest. If nothing at all has appeared after 10 minutes, press **Ctrl+C** and send the text in the terminal to the team.

### Step 6: Open the results

The result is a CSV file in the **`results`** folder inside the project folder, named after your run: `results/extracted_metadata_MY_FIRST_RUN.csv`. To get to it, either:

- **Click the link in the terminal:** when the run finishes, the terminal shows the path of the results file. Cmd-click it (Mac) or Ctrl-click it (Windows) to open it.
- **Use VS Code's file list:** open the `results` folder on the left and right-click the CSV → **Reveal in Finder** (Mac) / **Reveal in File Explorer** (Windows).
- **Open the folder from the terminal.** Mac:

  ```bash
  open results
  ```

  Windows:

  ```powershell
  explorer results
  ```

Open the CSV in Excel or any spreadsheet program. It has one row per drawing, with the extracted fields, a confidence score (0 to 1) for each field, the source image, the Node ID, and a `needs_manual_review` column marking rows to check by hand. See [Output files](#output-files) for every column.

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

Process a dataset in `data/`, referred to by folder name:

```bash
python run_pipeline.py --dataset_name YFA_cabana --run_name yfa_cabana
```

Quick check on the first N images only:

```bash
python run_pipeline.py --dataset_name YFA_cabana --run_name demo --limit 10
```

Process a dataset anywhere else (e.g. on the Desktop), referred to by path; same layout as in `data/`:

```bash
python run_pipeline.py --input_dir "/path/to/SomeFolder" --run_name some_batch
```

No arguments: processes `data/YFA_cabana`, `data/YFA_cabana_test` and `data/YFA_state_government` (missing folders are skipped with a warning):

```bash
python run_pipeline.py
```

No-TB branch: save a debug overlay per drawing (to check / tune the rules):

```bash
python run_pipeline.py --dataset_name Bates_smart_Mccutcheon --run_name bsm --limit 10 --no_tb_debug
```

Skip drawings without a title block (behaviour before the No-TB branch existed):

```bash
python run_pipeline.py --dataset_name YFA_cabana --run_name tb_only --no_tb skip
```

**General options**

| Option | Meaning |
|---|---|
| `--dataset_name <name>` | Process the dataset folder `data/<name>/` |
| `--input_dir <path>` | Process a dataset folder at any location (same layout: `metadata.csv` + one folder per Node ID) |
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
| `node_id` | Node ID of the drawing, from `metadata.csv` |
| `<field>`, `<field>_confidence` | Each field (`project_name`, `drawing_name`, `architect`, `draughtsperson`, `drawing_number`, `date`, `scale`, `drawing_types`) followed by its confidence from 0 to 1. Both are blank when the field was not found |
| `branch` | `tb` = read from the title block, `no_tb` = read from the whole sheet |
| `tb_score` | Title block detector score (below 0.5 → `no_tb`) |
| `drawing_types_source` | `keyword` = a type keyword was found in `drawing_name`; `unmatched` = none found |
| `needs_manual_review` | `TRUE` if `drawing_types` is `unmatched`, or if the row comes from the No-TB branch |

How to read them:

- Each row can be traced back to its drawing through `source_image` and `node_id`.
- Low confidence scores point to values worth checking by hand.
- `drawing_types` (Elevation / Perspective / Plan / Section / Detail) is derived from keywords in the extracted `drawing_name`. Its confidence of 1.000 means the keyword was found, not that `drawing_name` itself was read correctly: check `drawing_name_confidence` for that. If there is no keyword, the cell is left blank and should be filled in by hand.
- Rows with `branch = no_tb` come from a first version of the rule-based extractor and are always flagged for review (see [Known limitations](#known-limitations)).

---

## Troubleshooting

Problems during setup are covered in [Step 2](#step-2-check-that-setup-succeeded). This section covers everything else.

#### `ModuleNotFoundError` when running the pipeline

The environment is not active. Activate it as in [Step 3](#step-3-activate-the-environment); the terminal line should then start with `(venv)`.

#### `command not found` / `is not recognized` for `python`

Before the environment is activated, use `python3` on Mac/Linux and `py` on Windows. After activating ([Step 3](#step-3-activate-the-environment)), `python` works everywhere. If you just installed something, open a new terminal.

#### PowerShell refuses to run `Activate.ps1`

Run this once, then activate again:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

#### Model download fails with HTTP 404

GitHub reports a private repository's files as "not found" when you are not logged in or your account has no access. Make sure your account has been added as a collaborator, log in, and download again:

```bash
gh auth login
```

```bash
python download_models.py
```

Alternatively, set a token with read access to the repository before downloading. Mac / Linux:

```bash
export GITHUB_TOKEN=<token>
```

Windows PowerShell:

```powershell
$env:GITHUB_TOKEN="<token>"
```

Windows Command Prompt:

```bat
set GITHUB_TOKEN=<token>
```

#### Download the models by hand (firewall, no access)

See [If step 5 failed: download the models from GitHub](#if-step-5-failed-download-the-models-from-github).

#### `CERTIFICATE_VERIFY_FAILED` on Mac

Python installed from python.org cannot verify secure downloads until its certificates are installed (this can also stop the OCR weights from downloading on the first run). Open **Applications → Python 3.11** and double-click **Install Certificates.command**, then try again.

#### A drawing is missing from the CSV

Either OCR found no text on it (the terminal shows `OCR found no text, skipping`), or the run used `--no_tb skip` and the drawing is listed in `intermediate/<run_name>/no_tb_images.txt`. `intermediate/<run_name>/routing.csv` shows which branch each drawing went to.

#### A No-TB row has wrong or empty fields

Re-run that drawing with `--no_tb_debug` and open its overlay in `intermediate/<run_name>/no_tb_debug/`: blocks without a box were not read by OCR; a box with the wrong role means a rule needs adjusting (`src/no_tb/field_specs.py`).

#### `Warning: using model from old location 'output/...'`

The models are still in the old `output/` folder. Move them (Mac / Linux):

```bash
mkdir -p models && mv output/title_block_detector_v*.pth output/field_classifier_v* models/
```

On Windows, move `title_block_detector_v2.pth` and the `field_classifier_v2` folder from `output` to a new `models` folder in File Explorer.

#### Linux: `ImportError: libGL.so.1` when the pipeline starts

Minimal server images lack a graphics library that OpenCV loads (Debian/Ubuntu):

```bash
sudo apt install libgl1 libglib2.0-0
```

#### Linux server: long runs stop when the SSH session closes

Run the pipeline inside `tmux` or `screen`, or put `nohup` in front of the command.

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

### Updating a git clone

Developers working from a clone of the repository update with:

```bash
git checkout main
git pull
# with the environment activated:
pip install -r requirements.txt
python download_models.py
```

`download_models.py` only downloads models that are missing, so it is safe to run every time.

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