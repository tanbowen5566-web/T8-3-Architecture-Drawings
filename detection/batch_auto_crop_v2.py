"""
batch_auto_crop_v2.py

Purpose: Use the v2 detection model (trained on 51 images, stronger
         generalization) to batch-detect and crop title blocks from
         more raw drawings, producing more crop images for teammates
         to annotate for field classification.

Scope: By default scans YFA_cabana, YFA_cabana_test, and
       YFA_state_government (these styles all appear in the training
       data, so detection success rate is well supported).
       Bates_smart_Mccutcheon is NOT included by default -- these
       drawings may not have a standard title block region at all,
       which is a separate open problem; skipped here rather than
       forced through this pipeline.

Images with low confidence or no detection are skipped and logged,
without interrupting the batch. This makes it easy to come back later
and specifically address the "no title block detected" cases.

Usage:
    # Use the default three datasets (YFA_cabana, etc.)
    python batch_auto_crop_v2.py

    # Point at any folder (e.g. a new folder on the Desktop)
    python batch_auto_crop_v2.py --input_dir "/Users/dazai/Desktop/SomeFolder" --output_dir data/titleblock_crops_some_batch

    # Multiple input folders at once are also supported
    python batch_auto_crop_v2.py --input_dir folder1 folder2 --output_dir data/some_output_folder
"""

import argparse
import os

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image

MODEL_PATH = "output/title_block_detector_v2.pth"

# Defaults used when no command-line arguments are passed (backward compatible)
DEFAULT_SOURCE_DIRS = [
    "data/YFA_cabana",
    "data/YFA_cabana_test",
    "data/YFA_state_government",
]
DEFAULT_OUTPUT_DIR = "data/titleblock_crops_v2"
DEFAULT_FAILED_LOG_PATH = "output/batch_crop_v2_failed.txt"

NUM_CLASSES = 2
SCORE_THRESHOLD = 0.5  # Only trust high-confidence boxes for auto-cropping, to avoid cropping the wrong region
PADDING = 20


def parse_args():
    parser = argparse.ArgumentParser(description="Batch detect + crop title blocks")
    parser.add_argument(
        "--input_dir", nargs="+", default=DEFAULT_SOURCE_DIRS,
        help="One or more folders containing the raw images to process (project-relative path or any absolute path, e.g. a folder on the Desktop)",
    )
    parser.add_argument(
        "--output_dir", default=DEFAULT_OUTPUT_DIR,
        help="Folder to save the cropped results to",
    )
    parser.add_argument(
        "--failed_log", default=DEFAULT_FAILED_LOG_PATH,
        help="File to write the list of images that failed detection / had low confidence",
    )
    return parser.parse_args()


def build_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


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


def main():
    args = parse_args()
    device = torch.device("cpu")

    print("Loading v2 detection model...")
    model = build_model(NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.failed_log) or ".", exist_ok=True)

    images = find_all_images(args.input_dir)
    print(f"Found {len(images)} images to process\n")

    success_count = 0
    failed_images = []

    for i, (fname, img_path) in enumerate(images, 1):
        image = Image.open(img_path).convert("RGB")
        img_tensor = torchvision.transforms.functional.to_tensor(image).to(device)

        with torch.no_grad():
            prediction = model([img_tensor])[0]

        boxes = prediction["boxes"]
        scores = prediction["scores"]

        if len(scores) == 0 or scores.max() < SCORE_THRESHOLD:
            print(f"[{i}/{len(images)}] Skipped (no high-confidence title block detected): {fname}")
            failed_images.append(fname)
            continue

        best_idx = scores.argmax()
        x1, y1, x2, y2 = boxes[best_idx].tolist()
        score = scores[best_idx].item()

        x1 = max(0, x1 - PADDING)
        y1 = max(0, y1 - PADDING)
        x2 = min(image.width, x2 + PADDING)
        y2 = min(image.height, y2 + PADDING)

        cropped = image.crop((x1, y1, x2, y2))
        out_path = os.path.join(args.output_dir, fname)
        cropped.save(out_path)

        print(f"[{i}/{len(images)}] Success (confidence={score:.3f}): {fname}")
        success_count += 1

    with open(args.failed_log, "w", encoding="utf-8") as f:
        for fname in failed_images:
            f.write(fname + "\n")

    print(f"\nDone: {success_count} succeeded, {len(failed_images)} failed/skipped")
    print(f"Crops saved to: {args.output_dir}")
    print(f"Failed list saved to: {args.failed_log}")


if __name__ == "__main__":
    main()