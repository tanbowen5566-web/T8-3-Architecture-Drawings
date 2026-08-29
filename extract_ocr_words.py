"""
extract_ocr_words.py

用途：对title block裁剪图跑docTR OCR，提取每个词的文字+坐标，
      存成结构化JSON，供后续在Label Studio里做字段标注（LayoutLMv3训练数据准备）

注意：这一步不做任何字段抽取/分类，只负责"文字+坐标"的干净提取。
      字段分类交给后续训练的LayoutLMv3模型。

用法：
    python extract_ocr_words.py

依赖：
    pip install python-doctr[torch]
"""

import json
import os

from doctr.io import DocumentFile
from doctr.models import ocr_predictor

# 用自动裁剪的title block图（验证检测->裁剪->OCR完整pipeline）
INPUT_DIR = "data/titleblock_crops_auto"
OUTPUT_DIR = "data/ocr_words"


def load_ocr_model():
    print("加载docTR模型...")
    model = ocr_predictor(pretrained=True)
    print("模型加载完成。")
    return model


def extract_words_with_boxes(image_path, model):
    """跑docTR OCR，返回该图所有词的 {text, bbox} 列表。
    bbox是docTR原生的相对坐标 (x0, y0, x1, y1)，范围0-1，
    对应图片左上角为(0,0)，右下角为(1,1)。
    """
    doc = DocumentFile.from_images(str(image_path))
    result = model(doc)

    items = []
    for page in result.pages:
        for block in page.blocks:
            for line in block.lines:
                for word in line.words:
                    (x0, y0), (x1, y1) = word.geometry
                    items.append({
                        "text": word.value,
                        "confidence": round(float(word.confidence), 4),
                        "bbox": [round(x0, 4), round(y0, 4), round(x1, 4), round(y1, 4)],
                    })
    return items


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model = load_ocr_model()

    image_files = sorted(
        f for f in os.listdir(INPUT_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )

    if not image_files:
        print(f"警告：{INPUT_DIR} 里没有找到图片文件")
        return

    print(f"共 {len(image_files)} 张图待处理\n")

    for fname in image_files:
        image_path = os.path.join(INPUT_DIR, fname)
        print(f"处理: {fname}")

        try:
            words = extract_words_with_boxes(image_path, model)
        except Exception as e:
            print(f"  OCR失败: {e}")
            continue

        print(f"  提取到 {len(words)} 个词")

        out_name = os.path.splitext(fname)[0] + ".json"
        out_path = os.path.join(OUTPUT_DIR, out_name)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"source_image": fname, "words": words}, f, ensure_ascii=False, indent=2)

        print(f"  已保存: {out_path}\n")

    print("全部完成。")


if __name__ == "__main__":
    main()