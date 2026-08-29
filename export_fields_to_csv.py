"""
export_fields_to_csv.py

用途：对一批title block裁剪图跑"OCR + 字段分类"完整流程，
      把每张图的字段结果对接到队友的 src/input_output.py::save_output()，
      按团队约定的8列schema输出主CSV；
      另外把每个字段的预测置信度单独存一份文件（save_output不支持置信度列，
      不改队友的共用代码，避免影响其他模块）。

用法：
    python export_fields_to_csv.py

依赖：
    pip install python-doctr[torch] transformers torch pillow pandas
"""

import csv
import os

import torch
from PIL import Image
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification

from src.input_output import save_output

FIELD_MODEL_DIR = "output/field_classifier_v0"
IMAGE_DIR = "data/titleblock_crops_auto"
OUTPUT_CSV_PATH = "output/extracted_metadata.csv"
CONFIDENCE_CSV_PATH = "output/extracted_metadata_confidence.csv"

# 当前模型预测的字段（drawing_name / drawing_types 暂未纳入训练，
# 见 convert_annotations_to_layoutlm_format.py 里的说明，
# 队友的save_output()会自动把没出现的列留空，不需要我们额外处理）
MODEL_FIELDS = [
    "architect",
    "project_name",
    "draughtsperson",
    "date",
    "scale",
    "drawing_number",
]


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
    """对一张图的OCR词列表跑字段分类，返回：
    { field_name: {"value": "拼接后的文本", "confidence": 平均置信度} }
    没有被分到任何词的字段，value为空字符串，confidence为None。
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

    probs = torch.softmax(outputs.logits, dim=-1).squeeze(0)  # (seq_len, num_labels)
    predictions = probs.argmax(-1).tolist()
    word_ids = encoding.word_ids(batch_index=0)

    # 每个词取第一个token的预测标签+置信度
    word_predictions = {}  # word_idx -> (label, confidence)
    for token_idx, word_idx in enumerate(word_ids):
        if word_idx is None or word_idx in word_predictions:
            continue
        label = id2label[predictions[token_idx]]
        confidence = probs[token_idx][predictions[token_idx]].item()
        word_predictions[word_idx] = (label, confidence)

    # 按字段聚合：同一字段的多个词，文本拼接，置信度取平均
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


def main():
    os.makedirs(os.path.dirname(OUTPUT_CSV_PATH), exist_ok=True)

    print("加载OCR模型...")
    ocr_model = ocr_predictor(pretrained=True)

    print("加载字段分类模型...")
    processor = LayoutLMv3Processor.from_pretrained(FIELD_MODEL_DIR, apply_ocr=False)
    field_model = LayoutLMv3ForTokenClassification.from_pretrained(FIELD_MODEL_DIR)
    field_model.eval()

    image_files = sorted(
        f for f in os.listdir(IMAGE_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )

    if not image_files:
        print(f"警告：{IMAGE_DIR} 里没有找到图片")
        return

    print(f"共 {len(image_files)} 张图待处理\n")

    main_results = []       # 喂给队友的save_output()
    confidence_rows = []    # 单独存置信度

    for fname in image_files:
        print(f"处理: {fname}")
        image_path = os.path.join(IMAGE_DIR, fname)
        image = Image.open(image_path).convert("RGB")

        words_data = extract_words_with_boxes(image_path, ocr_model)
        if not words_data:
            print("  OCR没有提取到文字，跳过")
            continue

        fields = predict_fields(image, words_data, processor, field_model)

        # 队友save_output()要求的字段（drawing_name/drawing_types模型未预测，
        # 不填这两个key，save_output会自动留空）
        main_row = {field: fields[field]["value"] for field in MODEL_FIELDS}
        main_row["source_image"] = fname  # 额外列，方便核对，不在队友的8列schema里
        main_results.append(main_row)

        confidence_row = {"source_image": fname}
        for field in MODEL_FIELDS:
            conf = fields[field]["confidence"]
            confidence_row[f"{field}_confidence"] = f"{conf:.3f}" if conf is not None else ""
        confidence_rows.append(confidence_row)

        print("  完成")

    # 主CSV：交给队友的save_output()，严格按8列schema输出
    save_output(main_results, OUTPUT_CSV_PATH)

    # 置信度单独存一份，不影响队友的输出格式
    confidence_fieldnames = ["source_image"] + [f"{f}_confidence" for f in MODEL_FIELDS]
    with open(CONFIDENCE_CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=confidence_fieldnames)
        writer.writeheader()
        writer.writerows(confidence_rows)

    print(f"\n共处理 {len(main_results)} 张图")
    print(f"主输出（队友格式）: {OUTPUT_CSV_PATH}")
    print(f"置信度输出: {CONFIDENCE_CSV_PATH}")


if __name__ == "__main__":
    main()