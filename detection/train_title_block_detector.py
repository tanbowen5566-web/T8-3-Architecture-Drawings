"""
Title Block Detection - Faster R-CNN Fine-tuning Training Script
Purpose: Fine-tune a pretrained Faster R-CNN on 5 annotated images to learn
         to detect the title_block class

Usage:
    python train_title_block_detector.py

Dependencies:
    pip install torch torchvision pycocotools pillow
"""

import json
import os

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# ============================================================
# 1. Merged annotation data (produced by merge_title_block_annotations.py)
#    Updated 2026-08-30: merged the project-1-YFA + project-1-GOV batches,
#    51 unique images after dedup, from the YFA_cabana and
#    YFA_state_government sources
# ============================================================
MERGED_DATA_PATH = "data/merged_title_block_annotations.json"

# Classes: label 0 is conventionally reserved for background,
# our title_block starts at 1
# (torchvision Faster R-CNN convention: label 0 = background)
NUM_CLASSES = 2  # background + title_block


def get_source_group(image_path):
    """Determine the source group of an image by its path, used for
    stratified train/val splitting to ensure the validation set has
    representation from both sources (different visual styles), rather than
    pure random sampling that could end up drawing entirely from one
    style."""
    if "YFA_cabana" in image_path:
        return "YFA_cabana"
    if "YFA_state_government" in image_path:
        return "YFA_state_government"
    return "other"


# ============================================================
# 2. Dataset class
# ============================================================
class TitleBlockDataset(Dataset):
    def __init__(self, merged_data_path, transforms=None):
        with open(merged_data_path, encoding="utf-8") as f:
            merged = json.load(f)

        self.transforms = transforms
        # JSON keys become strings after deserialization, convert back to int here
        self.image_id_to_path = {
            int(k): v for k, v in merged["image_id_to_path"].items()
        }
        self.annotations_by_image = {
            ann["image_id"]: ann for ann in merged["annotations"]
        }
        self.image_ids = sorted(self.image_id_to_path.keys())

    def __len__(self):
        return len(self.image_ids)

    def get_source_groups(self):
        """Return the source group for each sample (in self.image_ids order),
        for use in stratified splitting"""
        return [
            get_source_group(self.image_id_to_path[img_id])
            for img_id in self.image_ids
        ]

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_path = self.image_id_to_path[img_id]
        image = Image.open(img_path).convert("RGB")

        ann = self.annotations_by_image[img_id]
        x, y, w, h = ann["bbox"]
        # torchvision expects [x_min, y_min, x_max, y_max]
        boxes = torch.as_tensor([[x, y, x + w, y + h]], dtype=torch.float32)
        # The merged data has no category_id (only title_block by default), label fixed at 1
        labels = torch.as_tensor([1], dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([img_id]),
            "area": (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]),
            "iscrowd": torch.zeros((1,), dtype=torch.int64),
        }

        if self.transforms:
            image = self.transforms(image)
        else:
            image = torchvision.transforms.functional.to_tensor(image)

        return image, target


def collate_fn(batch):
    return tuple(zip(*batch))


# ============================================================
# 3. Model: load pretrained Faster R-CNN, replace the classification head
# ============================================================
def build_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights="DEFAULT"  # COCO pretrained weights
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


# ============================================================
# 4. Main training loop
# ============================================================
def main():
    # Note: the MPS backend has known numerical instability issues with
    # Faster R-CNN's RoIAlign and other ops (commonly manifesting as loss
    # becoming nan). With such a small amount of data, it's more reliable to
    # just use CPU directly.
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    dataset = TitleBlockDataset(MERGED_DATA_PATH)
    print(f"Total dataset size: {len(dataset)} images")

    # Stratified split: sample separately from each source
    # (YFA_cabana / YFA_state_government), roughly 20% for validation, to
    # ensure both visual styles are represented in the validation set --
    # unlike pure random sampling, which could end up drawing entirely from
    # one style and skewing the validation loss
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
    print(f"Train set: {len(train_dataset)} images, Val set: {len(val_dataset)} images")

    train_loader = DataLoader(
        train_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn
    )

    model = build_model(NUM_CLASSES)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    num_epochs = 20  # With the dataset grown to 51 images, run a few more epochs

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for images, targets in train_loader:
            images = [img.to(device) for img in images]
            targets = [
                {k: v.to(device) for k, v in t.items()} for t in targets
            ]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            if not torch.isfinite(losses):
                print("Warning: loss is nan/inf, skipping this update. Loss breakdown:")
                for k, v in loss_dict.items():
                    print(f"  {k}: {v.item()}")
                optimizer.zero_grad()
                continue

            optimizer.zero_grad()
            losses.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()

            epoch_loss += losses.item()

        # Validation loss: torchvision's detection models only output
        # loss_dict in train() mode, so we still use train() mode here, but
        # wrap it in torch.no_grad() to avoid updating parameters -- this is
        # only a monitoring loss, it doesn't represent a standard validation
        # accuracy metric (like mAP); it's just used to roughly observe
        # whether the model is converging in a reasonable direction, rather
        # than simply memorizing the training set
        val_loss = 0.0
        with torch.no_grad():
            for images, targets in val_loader:
                images = [img.to(device) for img in images]
                targets = [
                    {k: v.to(device) for k, v in t.items()} for t in targets
                ]
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                if torch.isfinite(losses):
                    val_loss += losses.item()

        print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {epoch_loss:.4f}  Val Loss: {val_loss:.4f}")

    # Save model weights (v2: trained on 51 images, from the merged
    # YFA_cabana + YFA_state_government sources, distinct from the v0/v1
    # versions for easy comparison)
    os.makedirs("output", exist_ok=True)
    save_path = "models/title_block_detector_v2.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to: {save_path}")


if __name__ == "__main__":
    main()