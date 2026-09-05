"""
verify_field_classifier.py

Purpose: Load the trained LayoutLMv3 field classification model, run
         predictions on the crop images, output which field each word was
         predicted as, and compare against the ground-truth annotations to
         check whether the pipeline works (this does NOT evaluate
         generalization -- training and testing use the same batch of
         images)

Usage:
    python verify_field_classifier.py
"""

import json
import os

import torch
from PIL import Image, ImageDraw
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification

MODEL_DIR = "output/field_classifier_v0"
DATA_PATH = "data/layoutlm_training_data.json"
IMAGE_DIR = "data/titleblock_crops_auto"
OUTPUT_DIR = "output/field_predictions_visualized"

# One color per field, for visualization
LABEL_COLORS = {
    "architect": "#FF6B6B",
    "project_name": "#4ECDC4",
    "draughtsperson": "#FFA07A",
    "date": "#F7DC6F",
    "scale": "#BB8FCE",
    "drawing_number": "#98D8C8",
}


def normalize_bbox_0_1_to_1000(bbox):
    x0, y0, x1, y1 = bbox
    return [
        max(0, min(1000, int(x0 * 1000))),
        max(0, min(1000, int(y0 * 1000))),
        max(0, min(1000, int(x1 * 1000))),
        max(0, min(1000, int(y1 * 1000))),
    ]


def bbox_1000_to_pixels(bbox_1000, image_width, image_height):
    x0, y0, x1, y1 = bbox_1000
    return [
        x0 / 1000 * image_width,
        y0 / 1000 * image_height,
        x1 / 1000 * image_width,
        y1 / 1000 * image_height,
    ]


def main():
    device = torch.device("cpu")

    print("Loading model...")
    processor = LayoutLMv3Processor.from_pretrained(MODEL_DIR, apply_ocr=False)
    model = LayoutLMv3ForTokenClassification.from_pretrained(MODEL_DIR)
    model.to(device)
    model.eval()

    id2label = model.config.id2label

    with open(DATA_PATH, encoding="utf-8") as f:
        records = json.load(f)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    total_words = 0
    correct_words = 0

    for record in records:
        source_image = record["source_image"]
        image_path = os.path.join(IMAGE_DIR, source_image)
        image = Image.open(image_path).convert("RGB")

        words = [w["text"] for w in record["words"]]
        true_labels = [w["label"] for w in record["words"]]
        boxes = [normalize_bbox_0_1_to_1000(w["bbox"]) for w in record["words"]]

        encoding = processor(
            image,
            words,
            boxes=boxes,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt",
        )

        with torch.no_grad():
            outputs = model(**{k: v.to(device) for k, v in encoding.items()})

        predictions = outputs.logits.argmax(-1).squeeze().tolist()
        word_ids = encoding.word_ids(batch_index=0)

        # Each word may be split into multiple subword tokens; take the
        # prediction of the word's first token
        pred_labels_per_word = {}
        for token_idx, word_idx in enumerate(word_ids):
            if word_idx is None or word_idx in pred_labels_per_word:
                continue
            pred_labels_per_word[word_idx] = id2label[predictions[token_idx]]

        print(f"\n=== {source_image} ===")
        draw_image = image.copy()
        draw = ImageDraw.Draw(draw_image)

        for i, word in enumerate(words):
            true_label = true_labels[i]
            pred_label = pred_labels_per_word.get(i, "?")
            match = "✓" if pred_label == true_label else "✗"

            total_words += 1
            if pred_label == true_label:
                correct_words += 1

            if true_label != "O" or pred_label != "O":
                print(f"  {match} {word:20s} true={true_label:16s} predicted={pred_label}")

            if pred_label in LABEL_COLORS:
                px_box = bbox_1000_to_pixels(boxes[i], image.width, image.height)
                draw.rectangle(px_box, outline=LABEL_COLORS[pred_label], width=3)

        max_display_width = 900
        if draw_image.width > max_display_width:
            ratio = max_display_width / draw_image.width
            draw_image = draw_image.resize(
                (max_display_width, int(draw_image.height * ratio))
            )
        out_path = os.path.join(OUTPUT_DIR, source_image)
        draw_image.save(out_path)
        print(f"  Saved visualization: {out_path}")

    accuracy = correct_words / total_words if total_words else 0
    print(f"\nOverall word-level accuracy: {correct_words}/{total_words} = {accuracy:.2%}")
    print("(Note: this is measured on the same batch of images used for training, it only verifies pipeline correctness, not generalization ability)")


if __name__ == "__main__":
    main()