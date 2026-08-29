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
# 1. image_id -> 真实文件路径 映射表
#    （因为Label Studio导出的file_name是它内部的存储路径，
#     不是我们项目里的真实图片位置，所以手动对齐）
# ============================================================
IMAGE_ID_TO_PATH = {
    0: "data/YFA_cabana_test/455995/1238207_1984-0047-02638-00001.jpg",
    1: "data/YFA_cabana_test/455992/1238196_1984-0047-02635-00001.jpg",
    2: "data/YFA_cabana_test/455993/1238198_1984-0047-02636-00001.jpg",
    3: "data/YFA_cabana_test/455994/1238204_1984-0047-02637-00001.jpg",
    4: "data/YFA_cabana_test/455996/1238209_1984-0047-02639-00001.jpg",
}

COCO_JSON_PATH = "data/project-1-at-2026-08-29-08-25-a8279fd5/result.json"

# 类别：0 通常留给背景(background)，我们的title_block从1开始编号
# （torchvision的Faster R-CNN约定：label 0 = background）
NUM_CLASSES = 2  # background + title_block


# ============================================================
# 2. Dataset类
# ============================================================
class TitleBlockDataset(Dataset):
    def __init__(self, coco_json_path, image_id_to_path, transforms=None):
        with open(coco_json_path) as f:
            coco_data = json.load(f)

        self.transforms = transforms
        self.image_id_to_path = image_id_to_path

        # 按image_id把annotations分组（这里每张图只有1个框，但代码写成支持多个框）
        self.annotations_by_image = {}
        for ann in coco_data["annotations"]:
            img_id = ann["image_id"]
            self.annotations_by_image.setdefault(img_id, []).append(ann)

        # 保留有对应本地文件路径、且有标注的image_id
        self.image_ids = [
            img["id"]
            for img in coco_data["images"]
            if img["id"] in self.image_id_to_path
            and img["id"] in self.annotations_by_image
        ]

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_path = self.image_id_to_path[img_id]
        image = Image.open(img_path).convert("RGB")

        anns = self.annotations_by_image[img_id]

        boxes = []
        labels = []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            # torchvision要求 [x_min, y_min, x_max, y_max]
            boxes.append([x, y, x + w, y + h])
            # COCO里category_id=0 是title_block，我们训练时加1（0留给背景）
            labels.append(ann["category_id"] + 1)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([img_id]),
            "area": (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]),
            "iscrowd": torch.zeros((len(anns),), dtype=torch.int64),
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

    dataset = TitleBlockDataset(COCO_JSON_PATH, IMAGE_ID_TO_PATH)
    print(f"数据集大小: {len(dataset)} 张图")

    # 5张图太少，不划分train/val，全部用来训练（先跑通流程）
    data_loader = DataLoader(
        dataset, batch_size=1, shuffle=True, collate_fn=collate_fn
    )

    model = build_model(NUM_CLASSES)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    num_epochs = 10  # 数据少，先跑几轮看看loss是否下降

    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for images, targets in data_loader:
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

        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {epoch_loss:.4f}")

    # 保存模型权重
    os.makedirs("output", exist_ok=True)
    save_path = "output/title_block_detector_v0.pth"
    torch.save(model.state_dict(), save_path)
    print(f"模型已保存到: {save_path}")


if __name__ == "__main__":
    main()