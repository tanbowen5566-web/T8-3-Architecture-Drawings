"""
Title Block Detection - 泛化能力测试脚本
用途：用训练好的模型，对完全没见过的图片做预测，检验泛化能力

用法：
    python test_generalization.py
"""

import os

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image, ImageDraw

MODEL_PATH = "output/title_block_detector_v0.pth"
OUTPUT_DIR = "output/generalization_test"
NUM_CLASSES = 2
SCORE_THRESHOLD = 0.3

# 从未参与训练的图片，分两组：
#   同项目不同页（YFA_cabana，和训练用的YFA_cabana_test是同一批图纸的其他页）
#   不同项目（YFA_state_government，可能是同一家事务所但不同项目）
TEST_IMAGES = {
    "YFA_cabana_同项目其他页": [
        "data/YFA_cabana/456005/1238244_1984-0047-02648-00001.jpg",
        "data/YFA_cabana/456002/1238229_1984-0047-02645-00001.jpg",
    ],
    "YFA_state_government_不同项目": [
        "data/YFA_state_government/460658/1264434_1984-0047-02874-00001.jpg",
        "data/YFA_state_government/460667/1264443_1984-0047-02883-00001.jpg",
    ],
}


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

    for group_name, paths in TEST_IMAGES.items():
        group_dir = os.path.join(OUTPUT_DIR, group_name)
        os.makedirs(group_dir, exist_ok=True)
        print(f"\n===== {group_name} =====")

        for img_path in paths:
            if not os.path.exists(img_path):
                print(f"跳过（文件不存在）: {img_path}")
                continue

            image = Image.open(img_path).convert("RGB")
            img_tensor = torchvision.transforms.functional.to_tensor(image).to(device)

            with torch.no_grad():
                prediction = model([img_tensor])[0]

            boxes = prediction["boxes"]
            scores = prediction["scores"]

            draw_image = image.copy()
            draw = ImageDraw.Draw(draw_image)

            kept = 0
            for box, score in zip(boxes, scores):
                if score < SCORE_THRESHOLD:
                    continue
                kept += 1
                x1, y1, x2, y2 = box.tolist()
                line_width = max(5, int(image.width / 1000))
                draw.rectangle([x1, y1, x2, y2], outline="red", width=line_width)
                draw.text((x1, max(0, y1 - 60)), f"title_block {score:.2f}", fill="red")

            print(f"{os.path.basename(img_path)}: 检测到 {kept} 个框（阈值{SCORE_THRESHOLD}以上）")
            if len(scores) > 0:
                top_score = scores.max().item()
                print(f"   最高置信度: {top_score:.3f}")
            else:
                print("   模型没有输出任何候选框")

            max_display_width = 2000
            if draw_image.width > max_display_width:
                ratio = max_display_width / draw_image.width
                new_size = (max_display_width, int(draw_image.height * ratio))
                draw_image = draw_image.resize(new_size)

            out_path = os.path.join(group_dir, os.path.basename(img_path))
            draw_image.save(out_path)
            print(f"   已保存: {out_path}")


if __name__ == "__main__":
    main()