"""
export_fields_to_csv.py

Purpose: Run the full "OCR + field classification" pipeline on a batch
         of title block crop images, feed each image's field results
         into the teammate's src/input_output.py::save_output(),
         producing the main CSV in the team's agreed 8-column schema;
         also writes each field's prediction confidence to a separate
         file (save_output doesn't support a confidence column, and we
         don't want to modify the teammate's shared code / affect other
         modules).

2026-09-14 update: uses field_classifier_v1 (84-image training set)
instead of field_classifier_v0 (5-image proof of concept).

Usage:
    # Use the default path (data/titleblock_crops_auto)
    python export_fields_to_csv.py

    # Point at any crop folder and output filenames
    python export_fields_to_csv.py --image_dir data/titleblock_crops_folderA --output_csv output/extracted_metadata_folderA.csv

Dependencies:
    pip install python-doctr[torch] transformers torch pillow pandas
"""

import argparse
import csv
import os

import torch
from PIL import Image
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification

from src.input_output import save_output

FIELD_MODEL_DIR = "output/field_classifier_v1"

DEFAULT_IMAGE_DIR = "data/titleblock_crops_auto"
DEFAULT_OUTPUT_CSV_PATH = "output/extracted_metadata.csv"
DEFAULT_CONFIDENCE_CSV_PATH = "output/extracted_metadata_confidence.csv"

# Fields the current model predicts (drawing_name / drawing_types are not
# yet part of training -- see the notes in
# convert_annotations_to_layoutlm_format.py. The teammate's save_output()
# automatically leaves any missing column blank, so we don't need to
# handle that ourselves)
MODEL_FIELDS = [
    "architect",
    "project_name",
    "draughtsperson",
    "date",
    "scale",
    "drawing_number",
]


def parse_args():
    parser = argparse.ArgumentParser(description="OCR + field classification, output structured CSV")
    parser.add_argument(
        "--image_dir", default=DEFAULT_IMAGE_DIR,
        help="Folder containing the title block crop images to process (project-relative path or any absolute path)",
    )
    parser.add_argument(
        "--output_csv", default=DEFAULT_OUTPUT_CSV_PATH,
        help="Path for the main output CSV (teammate's 8-column schema)",
    )
    parser.add_argument(
        "--confidence_csv", default=DEFAULT_CONFIDENCE_CSV_PATH,
        help="Path for the confidence output CSV",
    )
    return parser.parse_args()


def normalize_bbox_0_1_to_1000(bbox):
    x0, y0, x1, y1 = bbox
    return [
        max(0, min(1000, int(x0 * 1000))),
        max(0, min(1000, int(y0 * 1000))),
        max(0, min(1000, int(x1 * 1000))),
        max(0, min(1000, int(y1 * 1000))),
    ]


def extract_words_with_boxes(image_path, ocr_model):
    doc = DocumentFile.from_images(str(image_path))
    result = ocr_model(doc)

    items = []
    for page in result.pages:
        for block in page.blocks:
            for line in block.lines:
                for word in line.words:
                    (x0, y0), (x1, y1) = word.geometry
                    items.append({"text": word.value, "bbox": [x0, y0, x1, y1]})
    return items


def predict_fields(image, words_data, processor, field_model):
    """Run field classification on one image's OCR word list. Returns:
    { field_name: {"value": "joined text", "confidence": average confidence} }
    Fields with no words assigned get value="" and confidence=None.
    """
    id2label = field_model.config.id2label

    words = [w["text"] for w in words_data]
    boxes = [normalize_bbox_0_1_to_1000(w["bbox"]) for w in words_data]

    encoding = processor(
        image, words, boxes=boxes,
        truncation=True, padding="max_length", max_length=512,
        return_tensors="pt",
    )

    with torch.no_grad():
        outputs = field_model(**encoding)

    probs = torch.softmax(outputs.logits, dim=-1).squeeze(0)  # (seq_len, num_labels)
    predictions = probs.argmax(-1).tolist()
    word_ids = encoding.word_ids(batch_index=0)

    word_predictions = {}
    for token_idx, word_idx in enumerate(word_ids):
        if word_idx is None or word_idx in word_predictions:
            continue
        label = id2label[predictions[token_idx]]
        confidence = probs[token_idx][predictions[token_idx]].item()
        word_predictions[word_idx] = (label, confidence)

    field_texts = {f: [] for f in MODEL_FIELDS}
    field_confidences = {f: [] for f in MODEL_FIELDS}

    for word_idx, word_text in enumerate(words):
        if word_idx not in word_predictions:
            continue
        label, confidence = word_predictions[word_idx]
        if label in MODEL_FIELDS:
            field_texts[label].append(word_text)
            field_confidences[label].append(confidence)

    result = {}
    for field in MODEL_FIELDS:
        if field_texts[field]:
            value = " ".join(field_texts[field])
            avg_confidence = sum(field_confidences[field]) / len(field_confidences[field])
        else:
            value = ""
            avg_confidence = None
        result[field] = {"value": value, "confidence": avg_confidence}

    return result


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)

    print("Loading OCR model...")
    ocr_model = ocr_predictor(pretrained=True)

    print("Loading field classification model (v1)...")
    processor = LayoutLMv3Processor.from_pretrained(FIELD_MODEL_DIR, apply_ocr=False)
    field_model = LayoutLMv3ForTokenClassification.from_pretrained(FIELD_MODEL_DIR)
    field_model.eval()

    image_files = sorted(
        f for f in os.listdir(args.image_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )

    if not image_files:
        print(f"Warning: no images found in {args.image_dir}")
        return

    print(f"{len(image_files)} images to process\n")

    main_results = []
    confidence_rows = []

    for fname in image_files:
        print(f"Processing: {fname}")
        image_path = os.path.join(args.image_dir, fname)
        image = Image.open(image_path).convert("RGB")

        words_data = extract_words_with_boxes(image_path, ocr_model)
        if not words_data:
            print("  OCR found no text, skipping")
            continue

        fields = predict_fields(image, words_data, processor, field_model)

        main_row = {field: fields[field]["value"] for field in MODEL_FIELDS}
        main_row["source_image"] = fname
        main_results.append(main_row)

        confidence_row = {"source_image": fname}
        for field in MODEL_FIELDS:
            conf = fields[field]["confidence"]
            confidence_row[f"{field}_confidence"] = f"{conf:.3f}" if conf is not None else ""
        confidence_rows.append(confidence_row)

        print("  Done")

    save_output(main_results, args.output_csv)

    confidence_fieldnames = ["source_image"] + [f"{f}_confidence" for f in MODEL_FIELDS]
    with open(args.confidence_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=confidence_fieldnames)
        writer.writeheader()
        writer.writerows(confidence_rows)

    print(f"\nProcessed {len(main_results)} images")
    print(f"Main output (teammate's format): {args.output_csv}")
    print(f"Confidence output: {args.confidence_csv}")


if __name__ == "__main__":
    main()