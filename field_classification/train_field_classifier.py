"""
train_field_classifier.py

Purpose: Fine-tune LayoutLMv3 to classify each word in a title block
         crop into one of the metadata fields (architect / project_name /
         draughtsperson / date / scale / drawing_number), or "O" if it
         doesn't belong to any of them.

2026-09-13 update: retrained on the larger 84-image annotation batch
(project-7), using a stratified train/val split by source dataset
(YFA_cabana vs YFA_state_government). The first run of this larger
dataset showed clear overfitting -- train loss kept falling smoothly
while val loss bottomed out around epoch 3 and then got worse for the
rest of training. So this version tracks the best validation loss seen
so far and saves that checkpoint, instead of just saving whatever the
last epoch happens to be.

Usage:
    python train_field_classifier.py

Dependencies:
    pip install transformers datasets torch pillow seqeval
"""

import json
import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import (
    LayoutLMv3Processor,
    LayoutLMv3ForTokenClassification,
)

SEED = 42  # fixes: train/val split order, DataLoader shuffling, and the
           # classifier head's random initialization -- all three were
           # previously uncontrolled, which is why two runs on the exact
           # same data/code produced very different training trajectories
           # (one learned reasonably, one got stuck and barely moved)

DATA_PATH = "data/layoutlm_training_data_v2.json"
IMAGE_DIR = "data/titleblock_crops_v2"  # the crops these annotations were made on

# Label set: O + 6 fields (drawing_name/drawing_types temporarily excluded,
# see the notes in convert_annotations_to_layoutlm_format.py)
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
    """LayoutLMv3 requires bboxes on a 0-1000 integer scale; we store
    0-1 relative coordinates, so just multiply by 1000 and round down."""
    x0, y0, x1, y1 = bbox
    return [
        max(0, min(1000, int(x0 * 1000))),
        max(0, min(1000, int(y0 * 1000))),
        max(0, min(1000, int(x1 * 1000))),
        max(0, min(1000, int(y1 * 1000))),
    ]


def get_source_group(source_image):
    """Classify each sample by which dataset it came from, for a
    stratified train/val split -- so validation reflects both visual
    styles rather than possibly landing entirely on one style by chance."""
    if "2018-0077" in source_image:
        return "YFA_state_government"
    if "1984-0047" in source_image:
        # Both YFA_cabana and (part of) YFA_state_government use this
        # accession prefix; YFA_cabana's node range is 02635-02650,
        # everything else with this prefix in this batch is
        # YFA_state_government.
        try:
            node_num = int(source_image.split("-")[2])
        except (IndexError, ValueError):
            return "other"
        if 2635 <= node_num <= 2650:
            return "YFA_cabana"
        return "YFA_state_government"
    return "other"


class FieldClassificationDataset(Dataset):
    def __init__(self, data_path, image_dir, processor):
        with open(data_path, encoding="utf-8") as f:
            self.records = json.load(f)
        self.image_dir = image_dir
        self.processor = processor

    def __len__(self):
        return len(self.records)

    def get_source_groups(self):
        return [get_source_group(r["source_image"]) for r in self.records]

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

        return {k: v.squeeze(0) for k, v in encoding.items()}


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device("cpu")  # small dataset, CPU is fine; also avoids MPS's known NaN issue with some ops
    print(f"Using device: {device}  (seed={SEED})")

    print("Loading LayoutLMv3 processor...")
    processor = LayoutLMv3Processor.from_pretrained(
        "microsoft/layoutlmv3-base", apply_ocr=False
    )

    dataset = FieldClassificationDataset(DATA_PATH, IMAGE_DIR, processor)
    print(f"Total dataset size: {len(dataset)} images")

    # Stratified split: hold out ~20% per source group for validation,
    # so both visual styles are represented in both splits
    source_groups = dataset.get_source_groups()
    group_to_indices = {}
    for idx, group in enumerate(source_groups):
        group_to_indices.setdefault(group, []).append(idx)

    rng = torch.Generator().manual_seed(42)
    train_indices = []
    val_indices = []
    for group, indices in group_to_indices.items():
        indices_tensor = torch.tensor(indices)
        shuffled = indices_tensor[torch.randperm(len(indices), generator=rng)]
        val_count = max(1, round(len(indices) * 0.2))
        val_indices.extend(shuffled[:val_count].tolist())
        train_indices.extend(shuffled[val_count:].tolist())
        print(f"  Source {group}: {len(indices)} total, {len(indices)-val_count} train, {val_count} val")

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    print(f"Train: {len(train_dataset)} images, Val: {len(val_dataset)} images")

    train_loader_generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, generator=train_loader_generator)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    print("Loading LayoutLMv3 model...")
    model = LayoutLMv3ForTokenClassification.from_pretrained(
        "microsoft/layoutlmv3-base",
        num_labels=len(LABEL_LIST),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    num_epochs = 15
    best_val_loss = float("inf")
    best_epoch = -1
    save_dir = "output/field_classifier_v1"

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
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

        # Validation loss -- LayoutLMv3ForTokenClassification computes loss
        # in eval() mode too as long as labels are passed, so this is a
        # proper eval-mode forward pass (dropout etc. disabled), unlike the
        # detection model which needed a train()-mode workaround for this.
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                if torch.isfinite(outputs.loss):
                    val_loss += outputs.loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {epoch_loss:.4f}  Val Loss: {val_loss:.4f}")

        # Track and save the checkpoint with the lowest validation loss
        # seen so far, rather than just whatever the final epoch is --
        # the first training run on this dataset showed val loss bottoming
        # out early (~epoch 3) and getting worse after that (overfitting),
        # so "last epoch" and "best epoch" are not the same thing here.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            os.makedirs(save_dir, exist_ok=True)
            model.save_pretrained(save_dir)
            processor.save_pretrained(save_dir)
            print(f"  -> New best val loss ({best_val_loss:.4f}), saved checkpoint")

    print(f"\nBest epoch: {best_epoch} (val loss {best_val_loss:.4f})")
    print(f"Model saved to: {save_dir}")


if __name__ == "__main__":
    main()