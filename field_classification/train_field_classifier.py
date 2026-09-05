"""
train_field_classifier.py

Purpose: Fine-tune LayoutLMv3 to do field-level token classification on
         title block crop images (which field each word belongs to:
         architect / project_name / draughtsperson / date / scale /
         drawing_number, or none of them, O)

Usage:
    python train_field_classifier.py

Dependencies:
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
IMAGE_DIR = "data/titleblock_crops_auto"  # same batch of crops used for OCR

# Label set: O + 6 fields (drawing_name/drawing_types temporarily excluded, see the conversion script's comments)
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
    """LayoutLMv3 requires bbox on a 0-1000 integer scale; we store 0-1
    relative coordinates, so just multiply by 1000 and truncate"""
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

        # Each tensor returned by the processor has a batch dimension (1, ...), squeeze it out
        return {k: v.squeeze(0) for k, v in encoding.items()}


def main():
    device = torch.device("cpu")  # small dataset, CPU is enough, avoids MPS's known numerical instability issues
    print(f"Using device: {device}")

    print("Loading LayoutLMv3 processor...")
    processor = LayoutLMv3Processor.from_pretrained(
        "microsoft/layoutlmv3-base", apply_ocr=False
    )

    dataset = FieldClassificationDataset(DATA_PATH, IMAGE_DIR, processor)
    print(f"Dataset size: {len(dataset)} images")

    data_loader = DataLoader(dataset, batch_size=1, shuffle=True)

    print("Loading LayoutLMv3 model...")
    model = LayoutLMv3ForTokenClassification.from_pretrained(
        "microsoft/layoutlmv3-base",
        num_labels=len(LABEL_LIST),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    num_epochs = 15  # small dataset, run more epochs, while watching whether loss keeps decreasing rather than oscillating

    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in data_loader:
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss

            if not torch.isfinite(loss):
                print("Warning: loss is nan/inf, skipping this update")
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
    print(f"\nModel saved to: {save_dir}")


if __name__ == "__main__":
    main()