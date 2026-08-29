"""
run_ocr_pipeline.py

Integrates:
  - src/input_output.py       (teammate's module: iter_images / save_output)
  - docTR                     (OCR text extraction, word-level + bounding boxes)
  - keyword-based field extraction (rule-based, no training required)

Usage:
    python run_ocr_pipeline.py <dataset_name> [limit]

Example:
    python run_ocr_pipeline.py YFA_cabana
    python run_ocr_pipeline.py YFA_cabana 3

What this script does:
  1. Iterate over every image in the given dataset using iter_images()
  2. Run docTR OCR on each image, keeping word-level text + bounding boxes
  3. Extract the 8 metadata fields using a mix of:
       - keyword-label matching (e.g. "DATE", "SCALES", "DRAWN") for
         fields that have an explicit printed label nearby
       - positional heuristics for fields with NO printed label at all
         (e.g. "architect" is just the company name block at the very
         top of the title block, with no "ARCHITECT:" label)
     This is a rule-based stopgap for the eventual LayoutLMv3
     field-classification model, used while training data is still
     being labelled by the team.
  4. Write results to output/extracted_metadata.csv via save_output()

Known limitations (see team notes):
  - Tuned against the "CABANA" (Yuncken Freeman) title block layout.
    Will likely need further keyword/threshold tuning for other
    drawing sets with different layouts.
  - "architect" assumes the architect/company name is always the
    first text block at the top of the title block. If a drawing's
    title block is laid out differently, this will need adjusting.
  - "date" specifically requires the matched value to contain a
    plausible year (4 digits, or 2 digits after an apostrophe) to
    avoid picking up the "DATE / PRINTED" row, which has no year.
"""

import re
import sys
from pathlib import Path

from doctr.io import DocumentFile
from doctr.models import ocr_predictor

# Import teammate's input/output module
from src.input_output import iter_images, save_output

# ---- Field extraction config ----
# Map: field name -> list of possible keyword variants that might
# appear on the drawing as a label for that field.
# Adjust these based on what you actually see on your drawings.
FIELD_KEYWORDS = {
    "drawing_number": ["drawing no", "drawing number", "dwg no", "dwg"],
    "date": ["date"],
    "scale": ["scale", "scales"],
    "draughtsperson": ["drawn", "drawn by", "draughtsperson", "draftsperson"],
}

# Fields handled by keyword matching above vs. fields handled by
# position-only heuristics (no reliable printed label exists).
# "architect", "project_name", "drawing_name" fall in this category:
# on the CABANA layout they are just plain text blocks stacked at
# the top of the title block, with no label word at all.

# Tolerance for "same line" / "same column" alignment when looking
# for a label's value nearby. These are relative (0-1) coordinates.
RIGHT_LINE_TOLERANCE = 0.02
BELOW_COLUMN_TOLERANCE = 0.15  # widened: label/value text widths differ a lot
RIGHT_SEARCH_MAX_GAP = 0.5     # widened: e.g. "SCALES" -> value far to the right

YEAR_PATTERN = re.compile(r"(19|20)\d{2}")


def load_ocr_model():
    print("Loading docTR model...")
    model = ocr_predictor(pretrained=True)
    print("Model loaded.")
    return model


def extract_words_with_boxes(image_path, model):
    """Run docTR OCR and return a flat list of {text, bbox} dicts.

    bbox is (x0, y0, x1, y1) in relative (0-1) coordinates, as
    returned natively by docTR. The list is ordered top-to-bottom,
    left-to-right (docTR's natural reading order), which we rely on
    for the positional heuristics below.
    """
    doc = DocumentFile.from_images(str(image_path))
    result = model(doc)

    items = []
    for page in result.pages:
        for block in page.blocks:
            for line in block.lines:
                for word in line.words:
                    (x0, y0), (x1, y1) = word.geometry
                    items.append({
                        "text": word.value,
                        "bbox": (x0, y0, x1, y1),
                    })
    return items


def normalize(text):
    """Lowercase and strip punctuation/colons for keyword matching."""
    return text.lower().strip().strip(":").strip()


def find_keyword_matches(words, field_keywords):
    """Return a list of (field_name, word_item) for every word that
    matches one of the known label keywords."""
    matches = []
    for field_name, keywords in field_keywords.items():
        for kw in keywords:
            for item in words:
                if normalize(item["text"]) == normalize(kw):
                    matches.append((field_name, item))
    return matches


def distance_to_right(label_bbox, candidate_bbox):
    """Distance metric: candidate is to the right of, and roughly
    level with, the label. Returns None if not plausible."""
    lx0, ly0, lx1, ly1 = label_bbox
    cx0, cy0, cx1, cy1 = candidate_bbox

    label_mid_y = (ly0 + ly1) / 2
    candidate_mid_y = (cy0 + cy1) / 2

    if abs(label_mid_y - candidate_mid_y) > RIGHT_LINE_TOLERANCE:
        return None
    if cx0 < lx1 - 0.005:
        return None
    if cx0 - lx1 > RIGHT_SEARCH_MAX_GAP:
        return None

    return cx0 - lx1


def distance_below(label_bbox, candidate_bbox):
    """Distance metric for a candidate below the label, in roughly
    the same column. Tolerance is wide because label/value text
    widths often differ a lot (e.g. "DRAWN" vs "H.v.L.")."""
    lx0, ly0, lx1, ly1 = label_bbox
    cx0, cy0, cx1, cy1 = candidate_bbox

    label_left = lx0
    candidate_left = cx0

    if abs(label_left - candidate_left) > BELOW_COLUMN_TOLERANCE:
        return None
    if cy0 < ly1 - 0.002:
        return None

    return cy0 - ly1


def find_candidates_for_label(label_item, all_words, max_candidates=6):
    """Return nearby words (right or below), sorted by distance,
    excluding the label itself and duplicate occurrences of the
    same label text."""
    label_bbox = label_item["bbox"]
    label_text_norm = normalize(label_item["text"])

    scored = []
    for item in all_words:
        if item is label_item:
            continue
        if normalize(item["text"]) == label_text_norm:
            continue

        d_right = distance_to_right(label_bbox, item["bbox"])
        d_below = distance_below(label_bbox, item["bbox"])

        d = None
        if d_right is not None:
            d = d_right
        if d_below is not None and (d is None or d_below < d):
            d = d_below

        if d is not None:
            scored.append((d, item))

    scored.sort(key=lambda pair: pair[0])
    return [item for _, item in scored[:max_candidates]]


def find_value_for_label(label_item, all_words, value_filter=None):
    """Find the nearest plausible value for a label. If value_filter
    is given, skip candidates that don't satisfy it (used for "date"
    to avoid matching non-date text like "PRINTED")."""
    candidates = find_candidates_for_label(label_item, all_words)

    for item in candidates:
        if value_filter is None or value_filter(item["text"]):
            return item["text"]

    return ""


def looks_like_date(text):
    return bool(YEAR_PATTERN.search(text))


def extract_architect_and_titles(words):
    """Position-only heuristic for fields with no printed label:
    architect / project_name / drawing_name are stacked text blocks
    at the top of the title block, separated by horizontal divider
    lines. We approximate "blocks" by grouping words into rows
    (similar y) and taking the first few rows as architect, the next
    few as project_name, then drawing_name — based on the CABANA
    layout structure.

    This is intentionally simple and WILL need adjusting per dataset;
    see module docstring.
    """
    if not words:
        return "", "", ""

    # Sort by vertical position (top to bottom), words already come
    # roughly in reading order from docTR, but sort defensively.
    sorted_words = sorted(words, key=lambda w: (w["bbox"][1], w["bbox"][0]))

    # Group into rows by y-position
    rows = []
    current_row = []
    current_y = None
    row_tolerance = 0.015

    for w in sorted_words:
        y = w["bbox"][1]
        if current_y is None or abs(y - current_y) <= row_tolerance:
            current_row.append(w)
            current_y = y if current_y is None else current_y
        else:
            rows.append(current_row)
            current_row = [w]
            current_y = y
    if current_row:
        rows.append(current_row)

    def row_text(row):
        row_sorted = sorted(row, key=lambda w: w["bbox"][0])
        return " ".join(w["text"] for w in row_sorted)

    row_texts = [row_text(r) for r in rows]

    # Heuristic based on CABANA layout: first 1-2 rows = architect
    # (company name), next 1-3 rows = project_name (until a row that
    # looks like a short all-caps sheet title), next row = drawing_name.
    # This is approximate — flag for manual review during testing.
    architect = " / ".join(row_texts[0:2]) if len(row_texts) >= 2 else (row_texts[0] if row_texts else "")
    project_name = " / ".join(row_texts[2:5]) if len(row_texts) >= 5 else ""
    drawing_name = row_texts[5] if len(row_texts) > 5 else ""

    return architect, project_name, drawing_name


def extract_fields(words):
    """Rule-based field extraction from a list of {text, bbox} words.

    This is a stopgap for LayoutLMv3 — swap this out once the model
    is trained on labelled data.
    """
    result = {field: "" for field in [
        "project_name", "drawing_name", "architect", "draughtsperson",
        "drawing_number", "date", "scale",
    ]}
    result["drawing_types"] = []

    # Fields with no printed label — position-based heuristic
    architect, project_name, drawing_name = extract_architect_and_titles(words)
    result["architect"] = architect
    result["project_name"] = project_name
    result["drawing_name"] = drawing_name

    # Fields with a printed label — keyword matching
    matches = find_keyword_matches(words, FIELD_KEYWORDS)

    for field_name, label_item in matches:
        if result.get(field_name):
            continue  # already filled by an earlier, better match

        value_filter = looks_like_date if field_name == "date" else None
        value = find_value_for_label(label_item, words, value_filter=value_filter)

        if value:
            result[field_name] = value

    return result


def run_pipeline(dataset_name, output_path="output/extracted_metadata.csv", limit=None):
    model = load_ocr_model()

    results = []
    count = 0

    for record in iter_images(dataset_name):
        image_path = record["image_path"]
        node_id = record["node_id"]

        print(f"\nProcessing: {image_path}")

        try:
            words = extract_words_with_boxes(image_path, model)
        except Exception as e:
            print(f"  Warning: OCR failed for {image_path} ({e})")
            words = []

        fields = extract_fields(words)

        print("  Extracted fields:")
        for k, v in fields.items():
            print(f"    {k}: {v}")

        result_row = dict(fields)
        result_row["node_id"] = node_id  # extra column, dropped by save_output()

        results.append(result_row)
        count += 1

        if limit is not None and count >= limit:
            print(f"\nReached limit of {limit} images, stopping.")
            break

    print(f"\nProcessed {count} image(s).")
    save_output(results, output_path)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_ocr_pipeline.py <dataset_name> [limit]")
        print("Example: python run_ocr_pipeline.py YFA_cabana 3")
        sys.exit(1)

    dataset_name = sys.argv[1]
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else None

    run_pipeline(dataset_name, limit=limit)