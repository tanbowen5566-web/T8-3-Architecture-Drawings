"""
test_end_to_end_new_image.py

Purpose: Run the full pipeline on a single image that never appeared in the
         training data at all:
         1. Locate and crop with the title block detection model
         2. Extract words + coordinates with docTR OCR
         3. Predict each word's field with the LayoutLMv3 field
            classification model

This is a more convincing check than the earlier "self-test on the same
batch of images" -- the image here is one the model has genuinely never
seen.

Usage:
    python test_end_to_end_new_image.py
"""

import os

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image, ImageDraw
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification

# The image the detection model successfully identified in the earlier
# generalization test (same project, different page, confidence 0.966)
TEST_IMAGE_PATH = "data/YFA_cabana/456002/1238229_1984-0047-02645-00001.jpg"

DETECTOR_MODEL_PATH = "output/title_block_detector_v0.pth"
FIELD_MODEL_DIR = "output/field_classifier_v0"
OUTPUT_DIR = "intermediate/end_to_end_test"
DETECTION_SCORE_THRESHOLD = 0.5
CROP_PADDING = 20

LABEL_COLORS = {
    "architect": "#FF6B6B",
    "project_name": "#4ECDC4",
    "draughtsperson": "#FFA07A",
    "date": "#F7DC6F",
    "scale": "#BB8FCE",
    "drawing_number": "#98D8C8",
}


def build_detector():
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    model.load_state_dict(torch.load(DETECTOR_MODEL_PATH, map_location="cpu"))
    model.eval()
    return model


def detect_and_crop(image_path, detector):
    image = Image.open(image_path).convert("RGB")
    img_tensor = torchvision.transforms.functional.to_tensor(image)

    with torch.no_grad():
        prediction = detector([img_tensor])[0]

    boxes = prediction["boxes"]
    scores = prediction["scores"]

    if len(scores) == 0 or scores.max() < DETECTION_SCORE_THRESHOLD:
        return None, None

    best_idx = scores.argmax()
    x1, y1, x2, y2 = boxes[best_idx].tolist()
    score = scores[best_idx].item()

    x1 = max(0, x1 - CROP_PADDING)
    y1 = max(0, y1 - CROP_PADDING)
    x2 = min(image.width, x2 + CROP_PADDING)
    y2 = min(image.height, y2 + CROP_PADDING)

    cropped = image.crop((x1, y1, x2, y2))
    return cropped, score


def run_ocr(cropped_image, ocr_model):
    doc = DocumentFile.from_images(
        [
            # docTR needs bytes when reading from an in-memory image; for
            # simplicity, write to disk first and read it back
        ]
    )
    # Simplified approach: save to a temp file first, then go through the
    # standard OCR interface, matching the OCR script logic written earlier
    tmp_path = os.path.join(OUTPUT_DIR, "_tmp_crop.jpg")
    cropped_image.save(tmp_path)

    doc = DocumentFile.from_images(tmp_path)
    result = ocr_model(doc)

    words = []
    for page in result.pages:
        for block in page.blocks:
            for line in block.lines:
                for word in line.words:
                    (x0, y0), (x1, y1) = word.geometry
                    words.append({"text": word.value, "bbox": [x0, y0, x1, y1]})

    os.remove(tmp_path)
    return words


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
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Test image: {TEST_IMAGE_PATH}\n")

    print("[1/3] Loading detection model, locating and cropping title block...")
    detector = build_detector()
    cropped, det_score = detect_and_crop(TEST_IMAGE_PATH, detector)

    if cropped is None:
        print("Detection failed: no title block found with sufficient confidence")
        return

    print(f"  Detection succeeded, confidence={det_score:.3f}")
    crop_save_path = os.path.join(OUTPUT_DIR, "cropped.jpg")
    cropped.save(crop_save_path)
    print(f"  Cropped image saved: {crop_save_path}")

    print("\n[2/3] Loading docTR, running OCR...")
    ocr_model = ocr_predictor(pretrained=True)
    words_data = run_ocr(cropped, ocr_model)
    print(f"  Extracted {len(words_data)} words")

    if not words_data:
        print("OCR extracted no text, aborting pipeline")
        return

    print("\n[3/3] Loading field classification model, predicting each word's field...")
    processor = LayoutLMv3Processor.from_pretrained(FIELD_MODEL_DIR, apply_ocr=False)
    field_model = LayoutLMv3ForTokenClassification.from_pretrained(FIELD_MODEL_DIR)
    field_model.eval()
    id2label = field_model.config.id2label

    words = [w["text"] for w in words_data]
    boxes = [normalize_bbox_0_1_to_1000(w["bbox"]) for w in words_data]

    encoding = processor(
        cropped, words, boxes=boxes,
        truncation=True, padding="max_length", max_length=512,
        return_tensors="pt",
    )

    with torch.no_grad():
        outputs = field_model(**encoding)

    predictions = outputs.logits.argmax(-1).squeeze().tolist()
    word_ids = encoding.word_ids(batch_index=0)

    pred_labels_per_word = {}
    for token_idx, word_idx in enumerate(word_ids):
        if word_idx is None or word_idx in pred_labels_per_word:
            continue
        pred_labels_per_word[word_idx] = id2label[predictions[token_idx]]

    print("\n===== Prediction results =====")
    draw_image = cropped.copy()
    draw = ImageDraw.Draw(draw_image)

    for i, word in enumerate(words):
        label = pred_labels_per_word.get(i, "O")
        if label != "O":
            print(f"  {label:16s} {word}")
        if label in LABEL_COLORS:
            px_box = bbox_1000_to_pixels(boxes[i], cropped.width, cropped.height)
            draw.rectangle(px_box, outline=LABEL_COLORS[label], width=3)

    out_path = os.path.join(OUTPUT_DIR, "prediction_visualized.jpg")
    draw_image.save(out_path)
    print(f"\nVisualization saved: {out_path}")
    print("\nNote: this image never appeared in the training data at all, so this result")
    print("reflects genuine generalization ability, not the earlier 100%-accuracy \"memorized the answers\" self-test.")


if __name__ == "__main__":
    main()