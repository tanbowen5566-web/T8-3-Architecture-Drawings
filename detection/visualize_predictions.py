"""
Title Block Detection - Inference Visualization Script
Purpose: Load the trained model, run predictions on images, draw the
         predicted boxes, and save the results

Usage:
    python visualize_predictions.py
"""

import os

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image, ImageDraw, ImageFont

# Same image paths as the training script
IMAGE_PATHS = [
    "data/YFA_cabana_test/455995/1238207_1984-0047-02638-00001.jpg",
    "data/YFA_cabana_test/455992/1238196_1984-0047-02635-00001.jpg",
    "data/YFA_cabana_test/455993/1238198_1984-0047-02636-00001.jpg",
    "data/YFA_cabana_test/455994/1238204_1984-0047-02637-00001.jpg",
    "data/YFA_cabana_test/455996/1238209_1984-0047-02639-00001.jpg",
]

MODEL_PATH = "output/title_block_detector_v0.pth"
OUTPUT_DIR = "output/predictions_visualized"
NUM_CLASSES = 2  # background + title_block

# Only draw boxes above this confidence threshold (set low for now, to see
# what the model is actually predicting)
SCORE_THRESHOLD = 0.3


def build_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def main():
    device = torch.device("cpu")  # Inference load is small, use CPU directly for consistency and stability with training

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

        # Draw boxes
        draw_image = image.copy()
        draw = ImageDraw.Draw(draw_image)

        kept = 0
        for box, score in zip(boxes, scores):
            if score < SCORE_THRESHOLD:
                continue
            kept += 1
            x1, y1, x2, y2 = box.tolist()
            # Scale up line width since the source images are large (10000+ px), thin lines would be hard to see
            line_width = max(5, int(image.width / 1000))
            draw.rectangle([x1, y1, x2, y2], outline="red", width=line_width)
            label_text = f"title_block {score:.2f}"
            draw.text((x1, max(0, y1 - 60)), label_text, fill="red")

        print(f"{os.path.basename(img_path)}: detected {kept} box(es) (above threshold {SCORE_THRESHOLD})")
        for box, score in zip(boxes, scores):
            print(f"   candidate box score={score:.3f} box={[round(v) for v in box.tolist()]}")

        # Save (downscaled a bit for easier viewing, since the original images are very large)
        max_display_width = 2000
        if draw_image.width > max_display_width:
            ratio = max_display_width / draw_image.width
            new_size = (max_display_width, int(draw_image.height * ratio))
            draw_image = draw_image.resize(new_size)

        out_path = os.path.join(OUTPUT_DIR, os.path.basename(img_path))
        draw_image.save(out_path)
        print(f"   Saved: {out_path}\n")


if __name__ == "__main__":
    main()