"""
extract_ocr_words.py

Purpose: Run docTR OCR on title block crop images, extract each word's text
         + coordinates, and store as structured JSON for later field
         annotation in Label Studio (preparing LayoutLMv3 training data).

Note: this step does not do any field extraction/classification, it's only
      responsible for a clean extraction of "text + coordinates". Field
      classification is handled later by the trained LayoutLMv3 model.

Usage:
    python extract_ocr_words.py

Dependencies:
    pip install python-doctr[torch]
"""

import json
import os

from doctr.io import DocumentFile
from doctr.models import ocr_predictor

# 85 title block images batch-cropped with the v2 detection model (second batch, larger scale)
INPUT_DIR = "data/titleblock_crops_v2"
OUTPUT_DIR = "data/ocr_words_v2"


def load_ocr_model():
    print("Loading docTR model...")
    model = ocr_predictor(pretrained=True)
    print("Model loaded.")
    return model


def extract_words_with_boxes(image_path, model):
    """Run docTR OCR, return a list of {text, bbox} for all words in the image.
    bbox is docTR's native relative coordinates (x0, y0, x1, y1), in the
    range 0-1, with (0,0) at the top-left of the image and (1,1) at the
    bottom-right.
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
                        "confidence": round(float(word.confidence), 4),
                        "bbox": [round(x0, 4), round(y0, 4), round(x1, 4), round(y1, 4)],
                    })
    return items


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model = load_ocr_model()

    image_files = sorted(
        f for f in os.listdir(INPUT_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )

    if not image_files:
        print(f"Warning: no image files found in {INPUT_DIR}")
        return

    print(f"{len(image_files)} images to process\n")

    for fname in image_files:
        image_path = os.path.join(INPUT_DIR, fname)
        print(f"Processing: {fname}")

        try:
            words = extract_words_with_boxes(image_path, model)
        except Exception as e:
            print(f"  OCR failed: {e}")
            continue

        print(f"  Extracted {len(words)} words")

        out_name = os.path.splitext(fname)[0] + ".json"
        out_path = os.path.join(OUTPUT_DIR, out_name)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"source_image": fname, "words": words}, f, ensure_ascii=False, indent=2)

        print(f"  Saved: {out_path}\n")

    print("All done.")


if __name__ == "__main__":
    main()