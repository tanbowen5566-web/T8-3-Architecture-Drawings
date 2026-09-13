"""
run_pipeline.py

Purpose: End-to-end entry point that chains together the two stages of
         the pipeline into a single command:
           1. Title block detection + crop (Faster R-CNN, v2 model)
           2. OCR + field classification -> structured CSV (docTR + LayoutLMv3)

2026-09-14 update: field classification now uses field_classifier_v1
(84-image training set, early-stopped at the best validation-loss
checkpoint) instead of field_classifier_v0 (5-image proof of concept).
On the held-out validation split, v1 gets 100% recall on architect,
96% on date, 91% on project_name; draughtsperson/drawing_number/scale
are weaker (62-70%), partly due to a handful of images with incomplete
annotations rather than purely model error -- see team notes.

Usage:
    # Dataset mode -- uses the team's iter_images(dataset_name), which
    # reads data/<dataset_name>/metadata.csv and each Node ID's folder.
    # This is the mode that gives Node ID traceability in the output.
    python run_pipeline.py --dataset_name YFA_cabana --run_name yfa_cabana

    # Quick demo run on just the first N images of a dataset
    python run_pipeline.py --dataset_name YFA_cabana --run_name demo --limit 10

    # Folder mode -- arbitrary folder, no metadata.csv required.
    # Traceability is by filename only (no Node ID available).
    python run_pipeline.py --input_dir "/Users/dazai/Desktop/SomeFolder" --run_name some_batch

    # Multiple input folders at once are also supported in folder mode
    python run_pipeline.py --input_dir folder1 folder2 --run_name some_batch

    # Run with no arguments to use the default datasets and run name (folder mode)
    python run_pipeline.py

This creates:
    data/titleblock_crops_<run_name>/            cropped title block images
    output/batch_crop_<run_name>_failed.txt       images where detection failed
    output/extracted_metadata_<run_name>.csv      main output (team's 8-column schema)
    output/extracted_metadata_<run_name>_confidence.csv   per-field confidence + Node ID
                                                   (Node ID column is only populated in
                                                   dataset mode; blank in folder mode)
"""

import argparse
import csv
import os

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification

from src.input_output import save_output, iter_images

DETECTOR_MODEL_PATH = "output/title_block_detector_v2.pth"
FIELD_MODEL_DIR = "output/field_classifier_v1"

DEFAULT_SOURCE_DIRS = [
    "data/YFA_cabana",
    "data/YFA_cabana_test",
    "data/YFA_state_government",
]
DEFAULT_RUN_NAME = "v2"

DETECTOR_NUM_CLASSES = 2
DETECTION_SCORE_THRESHOLD = 0.5  # Only trust high-confidence boxes for auto-cropping
CROP_PADDING = 20

MODEL_FIELDS = [
    "architect",
    "project_name",
    "draughtsperson",
    "date",
    "scale",
    "drawing_number",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the full pipeline: detect + crop title blocks, then OCR + classify fields into a CSV"
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--dataset_name",
        help="Name of a team dataset with a metadata.csv (e.g. YFA_cabana). Uses iter_images() so the output can be traced back to Node ID. Mutually exclusive with --input_dir.",
    )
    mode_group.add_argument(
        "--input_dir", nargs="+", default=None,
        help="One or more arbitrary folders of raw images (project-relative or absolute path, e.g. a folder on the Desktop). No metadata.csv required, but the output cannot be traced to a Node ID -- only by filename. Mutually exclusive with --dataset_name.",
    )
    parser.add_argument(
        "--run_name", default=DEFAULT_RUN_NAME,
        help="Short name used to build all output paths for this run (e.g. 'run_name=demo' produces data/titleblock_crops_demo, output/extracted_metadata_demo.csv, etc.)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Only process the first N images (useful for a quick demo run instead of the full dataset)",
    )
    args = parser.parse_args()
    if args.dataset_name is None and args.input_dir is None:
        args.input_dir = DEFAULT_SOURCE_DIRS
    return args


# ---------------------------------------------------------------------------
# Image discovery: either a plain folder scan, or the team's dataset format
# (metadata.csv + Node ID folders) via iter_images(), which additionally
# gives us Node ID traceability.
# ---------------------------------------------------------------------------

def find_all_images(source_dirs):
    """Scan all source directories and return a list of
    (real filename, full path). Duplicate filenames are only kept once
    (e.g. files that exist in both YFA_cabana_test and YFA_cabana)."""
    seen_names = set()
    result = []
    for base_dir in source_dirs:
        if not os.path.isdir(base_dir):
            print(f"Warning: directory not found, skipping: {base_dir}")
            continue
        for root, _dirs, files in os.walk(base_dir):
            for fname in sorted(files):
                if not fname.lower().endswith(".jpg"):
                    continue
                if fname in seen_names:
                    continue
                seen_names.add(fname)
                result.append((fname, os.path.join(root, fname)))
    return result


def get_images_and_node_ids(args):
    """Returns (images, node_id_by_filename).
    images is a list of (filename, full_path).
    node_id_by_filename maps filename -> Node ID string, and is only
    populated in dataset mode; empty dict in folder mode."""
    if args.dataset_name:
        print(f"Dataset mode: reading '{args.dataset_name}' via iter_images() (Node ID tracking enabled)\n")
        records = list(iter_images(args.dataset_name))
        images = [(r["image_name"], str(r["image_path"])) for r in records]
        node_id_by_filename = {r["image_name"]: r["node_id"] for r in records}
        return images, node_id_by_filename
    else:
        print("Folder mode: scanning provided folders (no Node ID tracking -- filename only)\n")
        images = find_all_images(args.input_dir)
        return images, {}


# ---------------------------------------------------------------------------
# Stage 1: detection + crop
# ---------------------------------------------------------------------------

def build_detector(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def detect_and_crop_all(images, crop_output_dir, failed_log_path, device):
    print("Loading title block detection model (v2)...")
    detector = build_detector(DETECTOR_NUM_CLASSES)
    detector.load_state_dict(torch.load(DETECTOR_MODEL_PATH, map_location=device))
    detector.to(device)
    detector.eval()

    os.makedirs(crop_output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(failed_log_path) or ".", exist_ok=True)

    print(f"Found {len(images)} images to process\n")

    success_count = 0
    failed_images = []

    for i, (fname, img_path) in enumerate(images, 1):
        image = Image.open(img_path).convert("RGB")
        img_tensor = torchvision.transforms.functional.to_tensor(image).to(device)

        with torch.no_grad():
            prediction = detector([img_tensor])[0]

        boxes = prediction["boxes"]
        scores = prediction["scores"]

        if len(scores) == 0 or scores.max() < DETECTION_SCORE_THRESHOLD:
            print(f"[{i}/{len(images)}] Skipped (no high-confidence title block detected): {fname}")
            failed_images.append(fname)
            continue

        best_idx = scores.argmax()
        x1, y1, x2, y2 = boxes[best_idx].tolist()
        score = scores[best_idx].item()

        x1 = max(0, x1 - CROP_PADDING)
        y1 = max(0, y1 - CROP_PADDING)
        x2 = min(image.width, x2 + CROP_PADDING)
        y2 = min(image.height, y2 + CROP_PADDING)

        cropped = image.crop((x1, y1, x2, y2))
        out_path = os.path.join(crop_output_dir, fname)
        cropped.save(out_path)

        print(f"[{i}/{len(images)}] Success (confidence={score:.3f}): {fname}")
        success_count += 1

    with open(failed_log_path, "w", encoding="utf-8") as f:
        for fname in failed_images:
            f.write(fname + "\n")

    print(f"\nDetection done: {success_count} succeeded, {len(failed_images)} failed/skipped")
    print(f"Crops saved to: {crop_output_dir}")
    print(f"Failed list saved to: {failed_log_path}\n")


# ---------------------------------------------------------------------------
# Stage 2: OCR + field classification -> CSV
# ---------------------------------------------------------------------------

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

    probs = torch.softmax(outputs.logits, dim=-1).squeeze(0)
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


def classify_fields_to_csv(crop_dir, output_csv_path, confidence_csv_path, node_id_by_filename):
    print("Loading OCR model...")
    ocr_model = ocr_predictor(pretrained=True)

    print("Loading field classification model (v1)...")
    processor = LayoutLMv3Processor.from_pretrained(FIELD_MODEL_DIR, apply_ocr=False)
    field_model = LayoutLMv3ForTokenClassification.from_pretrained(FIELD_MODEL_DIR)
    field_model.eval()

    os.makedirs(os.path.dirname(output_csv_path) or ".", exist_ok=True)

    image_files = sorted(
        f for f in os.listdir(crop_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )

    if not image_files:
        print(f"Warning: no images found in {crop_dir} -- nothing to classify")
        return

    print(f"{len(image_files)} images to process\n")

    main_results = []
    confidence_rows = []

    for fname in image_files:
        print(f"Processing: {fname}")
        image_path = os.path.join(crop_dir, fname)
        image = Image.open(image_path).convert("RGB")

        words_data = extract_words_with_boxes(image_path, ocr_model)
        if not words_data:
            print("  OCR found no text, skipping")
            continue

        fields = predict_fields(image, words_data, processor, field_model)

        # save_output() strictly filters to the team's 8-column schema, so
        # source_image / Node ID can't live in the main CSV -- they go in
        # the confidence CSV instead, which we fully control.
        main_row = {field: fields[field]["value"] for field in MODEL_FIELDS}
        main_results.append(main_row)

        confidence_row = {
            "source_image": fname,
            "node_id": node_id_by_filename.get(fname, ""),
        }
        for field in MODEL_FIELDS:
            conf = fields[field]["confidence"]
            confidence_row[f"{field}_confidence"] = f"{conf:.3f}" if conf is not None else ""
        confidence_rows.append(confidence_row)

        print("  Done")

    save_output(main_results, output_csv_path)

    confidence_fieldnames = ["source_image", "node_id"] + [f"{f}_confidence" for f in MODEL_FIELDS]
    with open(confidence_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=confidence_fieldnames)
        writer.writeheader()
        writer.writerows(confidence_rows)

    print(f"\nField classification done: {len(main_results)} images processed")
    print(f"Main output (team's schema): {output_csv_path}")
    print(f"Confidence + Node ID output: {confidence_csv_path}")
    if not node_id_by_filename:
        print("Note: running in folder mode -- 'node_id' column is blank. Use --dataset_name for Node ID traceability.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    device = torch.device("cpu")

    crop_output_dir = f"data/titleblock_crops_{args.run_name}"
    failed_log_path = f"output/batch_crop_{args.run_name}_failed.txt"
    output_csv_path = f"output/extracted_metadata_{args.run_name}.csv"
    confidence_csv_path = f"output/extracted_metadata_{args.run_name}_confidence.csv"

    images, node_id_by_filename = get_images_and_node_ids(args)

    if args.limit is not None:
        images = images[:args.limit]
        print(f"--limit {args.limit} applied: processing {len(images)} image(s)\n")

    print("===== Stage 1: title block detection + crop =====\n")
    detect_and_crop_all(images, crop_output_dir, failed_log_path, device)

    print("===== Stage 2: OCR + field classification =====\n")
    classify_fields_to_csv(crop_output_dir, output_csv_path, confidence_csv_path, node_id_by_filename)


if __name__ == "__main__":
    main()