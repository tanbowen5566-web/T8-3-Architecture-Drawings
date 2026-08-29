"""
test_end_to_end_new_image.py

用途：对一张完全没在训练数据里出现过的图，跑通完整流程：
      1. 用title block检测模型定位并裁剪
      2. 用docTR做OCR提取词+坐标
      3. 用LayoutLMv3字段分类模型预测每个词的字段

这是比之前"同批图片自测"更有说服力的验证——图片是模型完全没见过的。

用法：
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

# 之前泛化测试中，检测模型成功识别的那张图（同项目其他页，置信度0.966）
TEST_IMAGE_PATH = "data/YFA_cabana/456002/1238229_1984-0047-02645-00001.jpg"

DETECTOR_MODEL_PATH = "output/title_block_detector_v0.pth"
FIELD_MODEL_DIR = "output/field_classifier_v0"
OUTPUT_DIR = "output/end_to_end_test"
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
            # docTR从内存图片读取需要先转成bytes；简单起见先落盘再读
        ]
    )
    # 简化处理：先存临时文件再走标准OCR接口，兼容前面写的OCR脚本逻辑
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

    print(f"测试图片: {TEST_IMAGE_PATH}\n")

    print("[1/3] 加载检测模型，定位并裁剪title block...")
    detector = build_detector()
    cropped, det_score = detect_and_crop(TEST_IMAGE_PATH, detector)

    if cropped is None:
        print("检测失败：没有找到置信度足够高的title block")
        return

    print(f"  检测成功，置信度={det_score:.3f}")
    crop_save_path = os.path.join(OUTPUT_DIR, "cropped.jpg")
    cropped.save(crop_save_path)
    print(f"  裁剪图已保存: {crop_save_path}")

    print("\n[2/3] 加载docTR，做OCR...")
    ocr_model = ocr_predictor(pretrained=True)
    words_data = run_ocr(cropped, ocr_model)
    print(f"  提取到 {len(words_data)} 个词")

    if not words_data:
        print("OCR没有提取到任何文字，流程中止")
        return

    print("\n[3/3] 加载字段分类模型，预测每个词的字段...")
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

    print("\n===== 预测结果 =====")
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
    print(f"\n可视化结果已保存: {out_path}")
    print("\n注意：这张图完全没在训练数据里出现过，此结果代表真实泛化能力，")
    print("而不是之前100%准确率那种\"记住答案\"的自测。")


if __name__ == "__main__":
    main()