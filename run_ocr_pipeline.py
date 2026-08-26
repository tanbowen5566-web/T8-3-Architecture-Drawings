"""
run_ocr_pipeline.py

Integrates:
  - src/input_output.py  (teammate's module: iter_images / save_output)
  - docTR                (OCR text extraction)

Usage:
    python run_ocr_pipeline.py <dataset_name>

Example:
    python run_ocr_pipeline.py YFA_cabana

What this script currently does:
  1. Iterate over every image in the given dataset using iter_images()
  2. Run docTR OCR on each image to extract raw text
  3. Save a placeholder result row per image (the 8 metadata fields are
     still empty at this stage — field classification via LayoutLMv3
     comes later) plus the raw OCR text for inspection
  4. Write everything to output/extracted_metadata.csv via save_output()

This is an intermediate integration step: OCR extraction is wired up
end-to-end, but the field classification step (title block detection +
LayoutLMv3) is not yet implemented, so the 8 output fields are empty
placeholders for now. The raw OCR text is included as an extra column
so you can inspect what was extracted for each drawing.
"""

import sys
from pathlib import Path

from doctr.io import DocumentFile
from doctr.models import ocr_predictor

# Import teammate's input/output module
from src.input_output import iter_images, save_output


def load_ocr_model():
    print("Loading docTR model...")
    model = ocr_predictor(pretrained=True)
    print("Model loaded.")
    return model


def run_ocr_on_image(model, image_path):
    """Run docTR on a single image and return the extracted text as one string."""
    doc = DocumentFile.from_images(str(image_path))
    result = model(doc)

    words = []
    for page in result.pages:
        for block in page.blocks:
            for line in block.lines:
                for word in line.words:
                    words.append(word.value)

    return " ".join(words)


def run_pipeline(dataset_name, output_path="output/extracted_metadata.csv", limit=None):
    model = load_ocr_model()

    results = []
    count = 0

    for record in iter_images(dataset_name):
        image_path = record["image_path"]
        node_id = record["node_id"]

        print(f"\nProcessing: {image_path}")

        try:
            ocr_text = run_ocr_on_image(model, image_path)
        except Exception as e:
            print(f"  Warning: OCR failed for {image_path} ({e})")
            ocr_text = ""

        # Debug: print the raw OCR text directly, since save_output()
        # only keeps the 8 official metadata columns and will silently
        # drop this field otherwise.
        print(f"  OCR text ({len(ocr_text)} chars): {ocr_text[:200]}")

        # Placeholder result row.
        # The 8 required fields are still empty — field classification
        # (title block detection + LayoutLMv3) is a later pipeline stage.
        result_row = {
            "project_name": "",
            "drawing_name": "",
            "architect": "",
            "draughtsperson": "",
            "drawing_number": "",
            "date": "",
            "scale": "",
            "drawing_types": [],
            # Extra column, not part of the required output schema,
            # kept here for inspection during development.
            "node_id": node_id,
            "raw_ocr_text": ocr_text,
        }

        results.append(result_row)
        count += 1

        if limit is not None and count >= limit:
            print(f"\nReached limit of {limit} images, stopping.")
            break

    print(f"\nProcessed {count} image(s).")

    # Save a separate debug CSV that keeps node_id and raw_ocr_text —
    # save_output() only keeps the 8 official fields and would drop these.
    import pandas as pd
    debug_path = Path(output_path).parent / "debug_ocr_text.csv"
    debug_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results)[["node_id", "raw_ocr_text"]].to_csv(debug_path, index=False)
    print(f"Debug OCR text saved to: {debug_path}")

    save_output(results, output_path)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_ocr_pipeline.py <dataset_name> [limit]")
        print("Example: python run_ocr_pipeline.py YFA_cabana 3")
        sys.exit(1)

    dataset_name = sys.argv[1]
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else None

    run_pipeline(dataset_name, limit=limit)