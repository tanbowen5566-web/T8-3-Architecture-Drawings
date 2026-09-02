"""
merge_title_block_annotations.py

用途：合并多批Label Studio导出的title block检测标注（COCO格式），
      自动处理：
      1. 去重 —— 同一张图可能在不同批次、或同一批次内被标注多次
         （按真实文件名去重，不是按image_id，因为不同批次的image_id
          是各自独立编号的，不能直接比较）
      2. 路径解析 —— 自动在本地data/目录下搜索每个文件名对应的真实路径，
         不用像之前那样手动一条条对照
      3. 类别名纠错 —— 有的标注批次把"title_block"打成了"title_blcok"，
         统一纠正（本脚本只用bbox，类别名不影响逻辑，仅作说明）

用法：
    python merge_title_block_annotations.py

输出：
    data/merged_title_block_annotations.json
    格式：{"image_id_to_path": {...}, "annotations": [...]}
    每条annotation: {"image_id": int, "bbox": [x,y,w,h], "source_batch": str}
"""

import json
import os

# 待合并的标注批次（COCO格式，每个是一个含 images/ + result.json 的文件夹）
ANNOTATION_BATCHES = [
    "data/project-1-YFA",
    "data/project-1-GOV",
]

# 在这些目录下搜索原图真实路径（按需增减）
SEARCH_DIRS = [
    "data/YFA_cabana",
    "data/YFA_state_government",
    "data/Bates_smart_Mccutcheon",
]

OUTPUT_PATH = "data/merged_title_block_annotations.json"


def build_filename_to_path_index(search_dirs):
    """扫描本地目录，建立 真实文件名(不含路径) -> 完整路径 的索引"""
    index = {}
    for base_dir in search_dirs:
        if not os.path.isdir(base_dir):
            continue
        for root, _dirs, files in os.walk(base_dir):
            for fname in files:
                if fname.lower().endswith(".jpg"):
                    if fname in index:
                        # 同名文件出现在多个位置（比如YFA_cabana_test和YFA_cabana重复），
                        # 保留第一次找到的，不覆盖，避免路径来回跳
                        continue
                    index[fname] = os.path.join(root, fname)
    return index


def extract_real_filename(label_studio_file_name):
    """Label Studio导出的file_name形如：
    ..\\..\\label-studio\\media\\upload\\1\\54c8e959-1238196_1984-0047-02635-00001.jpg
    真实文件名是最后一段，去掉Label Studio加的uuid前缀（uuid-真实文件名）
    """
    basename = label_studio_file_name.replace("\\", "/").split("/")[-1]
    # uuid前缀形如 "54c8e959-"，用第一个"-"分割，但真实文件名本身也含"-"，
    # 所以只切一次，且uuid固定是8位hex+"-"
    if len(basename) > 9 and basename[8] == "-":
        return basename[9:]
    return basename


def main():
    print("扫描本地图片，建立文件名索引...")
    filename_index = build_filename_to_path_index(SEARCH_DIRS)
    print(f"共索引到 {len(filename_index)} 个唯一文件名\n")

    # 用真实文件名去重：{real_filename: {"bbox": [...], "source_batch": str}}
    merged_by_filename = {}
    skipped_duplicates = []
    unresolved = []

    for batch_dir in ANNOTATION_BATCHES:
        result_path = os.path.join(batch_dir, "result.json")
        if not os.path.exists(result_path):
            print(f"警告：找不到 {result_path}，跳过该批次")
            continue

        with open(result_path, encoding="utf-8") as f:
            data = json.load(f)

        # image_id -> 真实文件名
        id_to_realname = {
            img["id"]: extract_real_filename(img["file_name"])
            for img in data["images"]
        }

        # image_id -> bbox（假设每张图只有一个title_block框，取第一个）
        id_to_bbox = {}
        for ann in data["annotations"]:
            if ann["image_id"] not in id_to_bbox:
                id_to_bbox[ann["image_id"]] = ann["bbox"]

        for image_id, real_name in id_to_realname.items():
            if image_id not in id_to_bbox:
                continue
            bbox = id_to_bbox[image_id]

            if real_name in merged_by_filename:
                skipped_duplicates.append((real_name, batch_dir))
                continue

            merged_by_filename[real_name] = {
                "bbox": bbox,
                "source_batch": batch_dir,
            }

    print(f"合并后唯一图片数: {len(merged_by_filename)}")
    if skipped_duplicates:
        print(f"跳过的重复标注: {len(skipped_duplicates)} 条")
        for name, batch in skipped_duplicates:
            print(f"   {name}（来自 {batch}，已有更早批次的标注，保留先加入的那份）")

    # 解析真实路径
    image_id_to_path = {}
    annotations = []
    next_id = 0

    for real_name, info in sorted(merged_by_filename.items()):
        if real_name not in filename_index:
            unresolved.append(real_name)
            continue
        image_id_to_path[next_id] = filename_index[real_name]
        annotations.append({
            "image_id": next_id,
            "bbox": info["bbox"],
            "source_batch": info["source_batch"],
        })
        next_id += 1

    if unresolved:
        print(f"\n警告：{len(unresolved)} 个文件在本地找不到对应路径，已跳过：")
        for name in unresolved:
            print(f"   {name}")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {"image_id_to_path": image_id_to_path, "annotations": annotations},
            f, ensure_ascii=False, indent=2,
        )

    print(f"\n最终可用于训练的图片数: {len(image_id_to_path)}")
    print(f"已保存: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()