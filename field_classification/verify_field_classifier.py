"""
verify_field_classifier.py

Purpose: Evaluate the trained field classification model (v1) on the
         held-out validation set (17 images, seed=42, stratified split --
         same split used during training) that it never received gradient
         updates from. This is the real generalization check, unlike an
         earlier self-test that scored 100% because it reused training
         images.

Usage:
    python verify_field_classifier.py
"""

import json
import os

import torch
from PIL import Image, ImageDraw
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification

MODEL_DIR = "models/field_classifier_v2"
DATA_PATH = "data/layoutlm_training_data_v3.json"
IMAGE_DIR = "data/titleblock_crops_v2"
OUTPUT_DIR = "intermediate/field_predictions_v2_val_visualized"

LABEL_COLORS = {
    "architect": "#FF6B6B",
    "project_name": "#4ECDC4",
    "draughtsperson": "#FFA07A",
    "date": "#F7DC6F",
    "scale": "#BB8FCE",
    "drawing_number": "#98D8C8",
    "drawing_name": "#F1948A",
}

# The 17 validation images under seed=42 + stratified split (same logic as
# in train_field_classifier.py's get_source_group / split code)
VAL_IMAGES = [
    "1264412_1984-0047-02852-00001.jpg",
    "1232634_2018-0077-00820-00001.jpg",
    "1264574_2018-0077-00990-00001.jpg",
    "1232618_2018-0077-00812-00001.jpg",
    "1264446_1984-0047-02886-00001.jpg",
    "1264433_1984-0047-02873-00001.jpg",
    "1232642_2018-0077-00823-00001.jpg",
    "1232639_2018-0077-00822-00001.jpg",
    "1232667_2018-0077-00832-00001.jpg",
    "1264410_1984-0047-02850-00001.jpg",
    "1264428_1984-0047-02868-00001.jpg",
    "1264437_1984-0047-02877-00001.jpg",
    "1232626_2018-0077-00817-00001.jpg",
    "1232663_2018-0077-00831-00001.jpg",
    "1238223_1984-0047-02643-00001.jpg",
    "1238204_1984-0047-02637-00001.jpg",
    "1238225_1984-0047-02644-00001.jpg",
]


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
    return [x0 / 1000 * image_width, y0 / 1000 * image_height,
            x1 / 1000 * image_width, y1 / 1000 * image_height]


def main():
    device = torch.device("cpu")

    print("Loading model...")
    processor = LayoutLMv3Processor.from_pretrained(MODEL_DIR, apply_ocr=False)
    model = LayoutLMv3ForTokenClassification.from_pretrained(MODEL_DIR)
    model.to(device)
    model.eval()

    id2label = model.config.id2label

    with open(DATA_PATH, encoding="utf-8") as f:
        all_records = json.load(f)
    records_by_image = {r["source_image"]: r for r in all_records}

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    total_words = 0
    correct_words = 0
    # Per-field breakdown: how many words of each true label were
    # predicted correctly (only counting words that ARE that field,
    # not "O" words, since "O" dominates and would hide field-level detail)
    field_total = {}
    field_correct = {}

    for fname in VAL_IMAGES:
        record = records_by_image.get(fname)
        if record is None:
            print(f"Warning: {fname} not found in {DATA_PATH}, skipping")
            continue

        image_path = os.path.join(IMAGE_DIR, fname)
        image = Image.open(image_path).convert("RGB")

        words = [w["text"] for w in record["words"]]
        true_labels = [w["label"] for w in record["words"]]
        boxes = [normalize_bbox_0_1_to_1000(w["bbox"]) for w in record["words"]]

        encoding = processor(
            image, words, boxes=boxes,
            truncation=True, padding="max_length", max_length=512,
            return_tensors="pt",
        )

        with torch.no_grad():
            outputs = model(**{k: v.to(device) for k, v in encoding.items()})

        predictions = outputs.logits.argmax(-1).squeeze().tolist()
        word_ids = encoding.word_ids(batch_index=0)

        pred_labels_per_word = {}
        for token_idx, word_idx in enumerate(word_ids):
            if word_idx is None or word_idx in pred_labels_per_word:
                continue
            pred_labels_per_word[word_idx] = id2label[predictions[token_idx]]

        print(f"\n=== {fname} ===")
        draw_image = image.copy()
        draw = ImageDraw.Draw(draw_image)

        for i, word in enumerate(words):
            true_label = true_labels[i]
            pred_label = pred_labels_per_word.get(i, "?")
            match = "OK " if pred_label == true_label else "ERR"

            total_words += 1
            if pred_label == true_label:
                correct_words += 1

            if true_label != "O":
                field_total[true_label] = field_total.get(true_label, 0) + 1
                if pred_label == true_label:
                    field_correct[true_label] = field_correct.get(true_label, 0) + 1

            if true_label != "O" or pred_label != "O":
                print(f"  {match} {word:20s} true={true_label:16s} pred={pred_label}")

            if pred_label in LABEL_COLORS:
                px_box = bbox_1000_to_pixels(boxes[i], image.width, image.height)
                draw.rectangle(px_box, outline=LABEL_COLORS[pred_label], width=3)

        max_display_width = 900
        if draw_image.width > max_display_width:
            ratio = max_display_width / draw_image.width
            draw_image = draw_image.resize((max_display_width, int(draw_image.height * ratio)))
        out_path = os.path.join(OUTPUT_DIR, fname)
        draw_image.save(out_path)

    accuracy = correct_words / total_words if total_words else 0
    print(f"\n===== Overall =====")
    print(f"Word-level accuracy (including 'O'): {correct_words}/{total_words} = {accuracy:.2%}")

    print("\nPer-field recall (of words truly labelled with this field, how many predicted correctly):")
    for field in sorted(field_total.keys()):
        total = field_total[field]
        correct = field_correct.get(field, 0)
        print(f"  {field:16s} {correct}/{total} = {correct/total:.2%}")

    print("\nNote: this is measured on the held-out validation split -- these")
    print("17 images did not receive gradient updates during training, so")
    print("this reflects generalization, not memorization.")


if __name__ == "__main__":
    main()