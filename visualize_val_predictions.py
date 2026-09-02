"""
visualize_val_predictions.py

用途：加载v1检测模型（16张图训练，14训练+2验证划分），
      对验证集那2张图（模型训练时没有直接见过标注答案）跑预测并画框，
      用于肉眼检查模型是否真的学到了定位title block的能力。

用法：
    python visualize_val_predictions.py
"""

import os

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image, ImageDraw

MODEL_PATH = "output/title_block_detector_v2.pth"
OUTPUT_DIR = "output/val_predictions_visualized_v2"
NUM_CLASSES = 2
SCORE_THRESHOLD = 0.3

# 种子42、分层划分下，验证集是这10张图（7张YFA_state_government + 3张YFA_cabana）
VAL_IMAGES = [
    "data/YFA_state_government/453312/1232680_2018-0077-00837-00001.jpg",
    "data/YFA_state_government/460638/1264414_1984-0047-02854-00001.jpg",
    "data/YFA_state_government/460639/1264415_1984-0047-02855-00001.jpg",
    "data/YFA_state_government/453307/1232667_2018-0077-00832-00001.jpg",
    "data/YFA_state_government/453302/1232654_2018-0077-00827-00001.jpg",
    "data/YFA_state_government/453299/1232645_2018-0077-00824-00001.jpg",
    "data/YFA_state_government/453296/1232636_2018-0077-00821-00001.jpg",
    "data/YFA_cabana/455997/1238211_1984-0047-02640-00001.jpg",
    "data/YFA_cabana/455992/1238196_1984-0047-02635-00001.jpg",
    "data/YFA_cabana/456005/1238244_1984-0047-02648-00001.jpg",
]


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

    for img_path in VAL_IMAGES:
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
        for box, score in zip(boxes, scores):
            print(f"   候选框 score={score:.3f} box={[round(v) for v in box.tolist()]}")

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