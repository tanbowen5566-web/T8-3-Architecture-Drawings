# T8-3 Architecture Drawings

A Python-based project for processing and organising UMA (University of Melbourne Archives) architectural drawing records — automatically detecting title blocks, extracting text via OCR, and classifying the extracted text into structured metadata fields.

## Overview

This tool extracts key metadata (project name, drawing number, scale, drafter, architect, drawing name, and a unique identifier) from the title blocks of historical architectural drawings, outputting the results as CSV with per-field confidence scores. It is designed for archival cataloguing staff to reduce manual data entry.

**Constraint:** the entire pipeline must run fully offline, with no external AI APIs — all models run locally.

## Pipeline

| Stage | Tool | Purpose |
|---|---|---|
| 1. Title block detection | RT-DETR | Locate the title block region on each drawing |
| 2. Text recognition (OCR) | docTR | Extract text and bounding boxes from the title block |
| 3. Field classification | LayoutLMv3 | Classify extracted text into the 8 metadata fields |

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

### Run

```bash
python src/main.py
```

(Adjust the entry point above once the pipeline scripts are finalised.)

## Notes

- First run will download model weights (docTR, LayoutLMv3) — this requires an internet connection **once**. After that, the models are cached locally and can run fully offline.
- Do not commit the `venv/` folder or the `data/` folder — both are excluded via `.gitignore`.
- If you're contributing new dependencies, regenerate `requirements.txt` with:
  ```bash
  pip freeze > requirements.txt
  ```

## Project Structure

```
.
├── src/              # Source code
├── data/             # Drawing dataset (not tracked in git)
├── requirements.txt  # Python dependencies
├── setup.py          # Cross-platform environment setup script
└── README.md
```
A Python-based project for processing and organising UMA architectural drawing records.

## Dataset

The architectural drawing dataset was provided by the teaching team for this project.

The dataset is stored locally in the `data/` folder and is not included in this repository.
