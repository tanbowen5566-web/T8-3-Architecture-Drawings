"""
Title Block Detection - Auto-Crop Script
Purpose: Use the trained detection model to predict title block locations in
         images and automatically crop and save them.
         This replaces the previous "manually crop with Preview app" step,
         completing the detect -> crop pipeline.

Usage:
    python auto_crop_title_blocks.py
"""

import os

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image

MODEL_PATH = "output/title_block_detector_v0.pth"
OUTPUT_DIR = "data/titleblock_crops_auto"
NUM_CLASSES = 2
SCORE_THRESHOLD = 0.5  # Auto-crop only trusts high-confidence boxes, to avoid cropping the wrong region

# First use the current model to confirm it works on 5 images (same style,
# a drawing series the trained model has already seen).
# Once teammates annotate more data and the model is retrained, this can be
# swapped for the full image list.
IMAGE_PATHS = [
    "data/YFA_cabana_test/455995/1238207_1984-0047-02638-00001.jpg",
    "data/YFA_cabana_test/455992/1238196_1984-0047-02635-00001.jpg",
    "data/YFA_cabana_test/455993/1238198_1984-0047-02636-00001.jpg",
    "data/YFA_cabana_test/455994/1238204_1984-0047-02637-00001.jpg",
    "data/YFA_cabana_test/455996/1238209_1984-0047-02639-00001.jpg",
]

# Extra margin (in pixels) added around the predicted box when cropping, to
# avoid cropping too tightly and cutting off edge text
PADDING = 20


def build_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def main():
    device = torch.device("cpu")

    model = build_model(NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for img_path in IMAGE_PATHS:
        image = Image.open(img_path).convert("RGB")
        img_tensor = torchvision.transforms.functional.to_tensor(image).to(device)

        with torch.no_grad():
            prediction = model([img_tensor])[0]

        boxes = prediction["boxes"]
        scores = prediction["scores"]

        if len(scores) == 0 or scores.max() < SCORE_THRESHOLD:
            print(f"Skipped (no high-confidence detection): {os.path.basename(img_path)}")
            continue

        # Only take the highest-confidence box (there's only one title block per image)
        best_idx = scores.argmax()
        x1, y1, x2, y2 = boxes[best_idx].tolist()
        score = scores[best_idx].item()

        # Add padding, and make sure it doesn't go out of bounds
        x1 = max(0, x1 - PADDING)
        y1 = max(0, y1 - PADDING)
        x2 = min(image.width, x2 + PADDING)
        y2 = min(image.height, y2 + PADDING)

        cropped = image.crop((x1, y1, x2, y2))

        out_name = os.path.basename(img_path)
        out_path = os.path.join(OUTPUT_DIR, out_name)
        cropped.save(out_path)

        print(f"{out_name}: confidence={score:.3f}, crop box=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})")
        print(f"   Saved: {out_path}")


if __name__ == "__main__":
    main()