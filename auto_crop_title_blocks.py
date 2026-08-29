"""
Title Block Detection - 自动裁剪脚本
用途：用训练好的检测模型，对图片预测title block位置，并自动裁剪保存
      这是为了替代之前"手动用预览App裁剪"的步骤，跑通 检测->裁剪 这一环

用法：
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
SCORE_THRESHOLD = 0.5  # 自动裁剪只信任高置信度的框，避免裁到错误区域

# 先用当前模型确认能work的5张图（同风格，训练模型见过的图纸系列）
# 等组员标注更多数据、重新训练模型之后，这里可以换成完整的图片列表
IMAGE_PATHS = [
    "data/YFA_cabana_test/455995/1238207_1984-0047-02638-00001.jpg",
    "data/YFA_cabana_test/455992/1238196_1984-0047-02635-00001.jpg",
    "data/YFA_cabana_test/455993/1238198_1984-0047-02636-00001.jpg",
    "data/YFA_cabana_test/455994/1238204_1984-0047-02637-00001.jpg",
    "data/YFA_cabana_test/455996/1238209_1984-0047-02639-00001.jpg",
]

# 裁剪时在预测框基础上留一点边距（像素），避免裁得太紧切掉边缘文字
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
            print(f"跳过（没有高置信度检测结果）: {os.path.basename(img_path)}")
            continue

        # 只取置信度最高的那一个框（title block在一张图里只有一个）
        best_idx = scores.argmax()
        x1, y1, x2, y2 = boxes[best_idx].tolist()
        score = scores[best_idx].item()

        # 加padding，并确保不越界
        x1 = max(0, x1 - PADDING)
        y1 = max(0, y1 - PADDING)
        x2 = min(image.width, x2 + PADDING)
        y2 = min(image.height, y2 + PADDING)

        cropped = image.crop((x1, y1, x2, y2))

        out_name = os.path.basename(img_path)
        out_path = os.path.join(OUTPUT_DIR, out_name)
        cropped.save(out_path)

        print(f"{out_name}: 置信度={score:.3f}，裁剪框=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})")
        print(f"   已保存: {out_path}")


if __name__ == "__main__":
    main()