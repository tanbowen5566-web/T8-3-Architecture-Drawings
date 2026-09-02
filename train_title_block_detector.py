"""
Title Block Detection - Faster R-CNN 微调训练脚本
用途：在5张标注图上微调一个预训练的Faster R-CNN，学会检测title_block这个类别

用法：
    python train_title_block_detector.py

依赖：
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
# 1. 合并后的标注数据（由 merge_title_block_annotations.py 生成）
#    2026-08-30更新：合并了project-1-YFA + project-1-GOV两批标注，
#    去重后共51张唯一图片，来自YFA_cabana和YFA_state_government两个来源
# ============================================================
MERGED_DATA_PATH = "data/merged_title_block_annotations.json"

# 类别：0 通常留给背景(background)，我们的title_block从1开始编号
# （torchvision的Faster R-CNN约定：label 0 = background）
NUM_CLASSES = 2  # background + title_block


def get_source_group(image_path):
    """按图片路径判断数据来源分组，用于分层划分train/val，
    保证验证集里两种来源（不同视觉风格）都有代表，而不是纯随机抽样
    可能全抽到同一种风格。"""
    if "YFA_cabana" in image_path:
        return "YFA_cabana"
    if "YFA_state_government" in image_path:
        return "YFA_state_government"
    return "other"


# ============================================================
# 2. Dataset类
# ============================================================
class TitleBlockDataset(Dataset):
    def __init__(self, merged_data_path, transforms=None):
        with open(merged_data_path, encoding="utf-8") as f:
            merged = json.load(f)

        self.transforms = transforms
        # json的key在反序列化后是字符串，这里转回int
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
        """返回每个样本（按self.image_ids顺序）的来源分组，供分层划分使用"""
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
        # torchvision要求 [x_min, y_min, x_max, y_max]
        boxes = torch.as_tensor([[x, y, x + w, y + h]], dtype=torch.float32)
        # 合并数据里没有category_id（默认只有title_block这一类），标签固定为1
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
# 3. 模型：加载预训练Faster R-CNN，替换分类头
# ============================================================
def build_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights="DEFAULT"  # COCO预训练权重
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


# ============================================================
# 4. 训练主流程
# ============================================================
def main():
    # 注意：MPS后端在Faster R-CNN的RoIAlign等算子上有已知的数值不稳定问题
    # （常见表现就是loss变成nan），数据量很小时直接用CPU更稳妥
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print(f"使用设备: {device}")

    dataset = TitleBlockDataset(MERGED_DATA_PATH)
    print(f"数据集总大小: {len(dataset)} 张图")

    # 分层划分：按数据来源（YFA_cabana / YFA_state_government）分别抽取
    # 约20%做验证集，保证验证集里两种视觉风格都有代表，
    # 不会像纯随机抽样那样可能全抽到同一种风格，导致验证loss失真
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
        print(f"  来源 {group}: 共{len(indices)}张，训练{len(indices)-val_count}张，验证{val_count}张")

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    print(f"训练集: {len(train_dataset)} 张，验证集: {len(val_dataset)} 张")

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

    num_epochs = 20  # 数据量增加到51张，适当再多跑几轮

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
                print("警告：loss为nan/inf，本次跳过更新。各分项loss：")
                for k, v in loss_dict.items():
                    print(f"  {k}: {v.item()}")
                optimizer.zero_grad()
                continue

            optimizer.zero_grad()
            losses.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()

            epoch_loss += losses.item()

        # 验证集loss：torchvision的detection模型在train()模式下才输出loss_dict，
        # 所以这里仍用train()模式，但用torch.no_grad()避免更新参数——
        # 这只是监控用的loss，不代表标准的验证集精度指标（如mAP），
        # 只用来大致观察模型是否在往合理方向收敛，而非单纯记住训练集
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

    # 保存模型权重（v2：51张图训练，来自YFA_cabana+YFA_state_government
    # 两个来源合并后的数据，区别于v0/v1版本，方便对比）
    os.makedirs("output", exist_ok=True)
    save_path = "output/title_block_detector_v2.pth"
    torch.save(model.state_dict(), save_path)
    print(f"模型已保存到: {save_path}")


if __name__ == "__main__":
    main()