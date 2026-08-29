"""
train_field_classifier.py

用途：微调LayoutLMv3，对title block裁剪图做字段级token分类
      （每个词属于哪个字段：architect / project_name / draughtsperson /
       date / scale / drawing_number，或者不属于任何字段 O）

用法：
    python train_field_classifier.py

依赖：
    pip install transformers datasets torch pillow seqeval
"""

import json
import os

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import (
    LayoutLMv3Processor,
    LayoutLMv3ForTokenClassification,
)

DATA_PATH = "data/layoutlm_training_data.json"
IMAGE_DIR = "data/titleblock_crops_auto"  # 和OCR用的同一批裁剪图

# 标签集合：O + 6个字段（drawing_name/drawing_types 暂时排除，见转换脚本注释）
LABEL_LIST = [
    "O",
    "architect",
    "project_name",
    "draughtsperson",
    "date",
    "scale",
    "drawing_number",
]
LABEL2ID = {label: i for i, label in enumerate(LABEL_LIST)}
ID2LABEL = {i: label for i, label in enumerate(LABEL_LIST)}


def normalize_bbox_0_1_to_1000(bbox):
    """LayoutLMv3要求bbox是0-1000整数刻度，我们存的是0-1相对坐标，直接乘1000取整"""
    x0, y0, x1, y1 = bbox
    return [
        max(0, min(1000, int(x0 * 1000))),
        max(0, min(1000, int(y0 * 1000))),
        max(0, min(1000, int(x1 * 1000))),
        max(0, min(1000, int(y1 * 1000))),
    ]


class FieldClassificationDataset(Dataset):
    def __init__(self, data_path, image_dir, processor):
        with open(data_path, encoding="utf-8") as f:
            self.records = json.load(f)
        self.image_dir = image_dir
        self.processor = processor

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        image_path = os.path.join(self.image_dir, record["source_image"])
        image = Image.open(image_path).convert("RGB")

        words = [w["text"] for w in record["words"]]
        boxes = [normalize_bbox_0_1_to_1000(w["bbox"]) for w in record["words"]]
        labels = [LABEL2ID[w["label"]] for w in record["words"]]

        encoding = self.processor(
            image,
            words,
            boxes=boxes,
            word_labels=labels,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt",
        )

        # processor返回的每个tensor都带batch维度(1, ...)，squeeze掉
        return {k: v.squeeze(0) for k, v in encoding.items()}


def main():
    device = torch.device("cpu")  # 数据量小，CPU即可，避免MPS已知的数值不稳定问题
    print(f"使用设备: {device}")

    print("加载LayoutLMv3 processor...")
    processor = LayoutLMv3Processor.from_pretrained(
        "microsoft/layoutlmv3-base", apply_ocr=False
    )

    dataset = FieldClassificationDataset(DATA_PATH, IMAGE_DIR, processor)
    print(f"数据集大小: {len(dataset)} 张图")

    data_loader = DataLoader(dataset, batch_size=1, shuffle=True)

    print("加载LayoutLMv3模型...")
    model = LayoutLMv3ForTokenClassification.from_pretrained(
        "microsoft/layoutlmv3-base",
        num_labels=len(LABEL_LIST),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    num_epochs = 15  # 数据量小，多跑几轮，同时留意loss是否持续下降而非震荡

    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in data_loader:
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss

            if not torch.isfinite(loss):
                print("警告：loss为nan/inf，跳过本次更新")
                optimizer.zero_grad()
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {epoch_loss:.4f}")

    os.makedirs("output", exist_ok=True)
    save_dir = "output/field_classifier_v0"
    model.save_pretrained(save_dir)
    processor.save_pretrained(save_dir)
    print(f"\n模型已保存到: {save_dir}")


if __name__ == "__main__":
    main()