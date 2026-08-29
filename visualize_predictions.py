"""
Title Block Detection - 推理可视化脚本
用途：加载训练好的模型，对图片做预测，画出预测框并保存

用法：
    python visualize_predictions.py
"""

import os

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image, ImageDraw, ImageFont

# 和训练脚本保持一致的图片路径
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

# 只画置信度高于这个阈值的框（先设低一点，方便看模型到底预测了什么）
SCORE_THRESHOLD = 0.3


def build_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def main():
    device = torch.device("cpu")  # 推理量小，直接用CPU，稳妥且和训练一致

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

        # 画框
        draw_image = image.copy()
        draw = ImageDraw.Draw(draw_image)

        kept = 0
        for box, score in zip(boxes, scores):
            if score < SCORE_THRESHOLD:
                continue
            kept += 1
            x1, y1, x2, y2 = box.tolist()
            # 线宽按图片尺寸放大一些，因为原图很大（1万多像素），线太细看不清
            line_width = max(5, int(image.width / 1000))
            draw.rectangle([x1, y1, x2, y2], outline="red", width=line_width)
            label_text = f"title_block {score:.2f}"
            draw.text((x1, max(0, y1 - 60)), label_text, fill="red")

        print(f"{os.path.basename(img_path)}: 检测到 {kept} 个框 (阈值{SCORE_THRESHOLD}以上)")
        for box, score in zip(boxes, scores):
            print(f"   候选框 score={score:.3f} box={[round(v) for v in box.tolist()]}")

        # 保存（缩小一点方便打开查看，原图太大）
        max_display_width = 2000
        if draw_image.width > max_display_width:
            ratio = max_display_width / draw_image.width
            new_size = (max_display_width, int(draw_image.height * ratio))
            draw_image = draw_image.resize(new_size)

        out_path = os.path.join(OUTPUT_DIR, os.path.basename(img_path))
        draw_image.save(out_path)
        print(f"   已保存: {out_path}\n")


if __name__ == "__main__":
    main()