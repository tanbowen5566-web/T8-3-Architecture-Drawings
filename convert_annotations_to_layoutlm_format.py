"""
convert_annotations_to_layoutlm_format.py

用途：把Label Studio导出的字段分类标注（字符范围+标签），
      对应回每个词（通过char_map），生成LayoutLMv3训练需要的格式：
      每张图 -> 词列表，每个词带 text / bbox / label

标注是字符级的（start/end基于拼接文本），char_map记录了每个词
在拼接文本中的字符范围。这里用"重叠"判断把标签分配给对应的词：
如果一个词的字符范围和某个标注span有重叠，就把该span的标签
赋给这个词。没有被任何span覆盖的词，标签记为 "O"（NER标准写法，
表示"不属于任何目标字段"）。

用法：
    python convert_annotations_to_layoutlm_format.py

输入：
    data/field_annotations/project-2-at-2026-08-29-12-04-a62de1c8.json
    data/ocr_words_char_map/*.json

输出：
    data/layoutlm_training_data.json
"""

import json
import os

ANNOTATIONS_PATH = "data/field_annotations/project-2-at-2026-08-29-12-04-a62de1c8.json"
CHAR_MAP_DIR = "data/ocr_words_char_map"
OUTPUT_PATH = "data/layoutlm_training_data.json"

# 暂时排除的字段：drawing_name 和 drawing_types 在标注时被标在了同一段
# 文字上（重叠span），而LayoutLMv3的token分类要求每个词只能有一个标签，
# 两者定义上的边界还没和团队确认清楚，先跳过，把pipeline跑通。
# 等字段定义确认后，把下面这行清空或调整即可重新纳入。
EXCLUDED_FIELDS = {"drawing_name", "drawing_types"}


def spans_overlap(a_start, a_end, b_start, b_end):
    return a_start < b_end and b_start < a_end


def load_char_map_for_image(source_image):
    """char_map文件名规则：<image_stem>_char_map.json"""
    stem = os.path.splitext(source_image)[0]
    map_path = os.path.join(CHAR_MAP_DIR, f"{stem}_char_map.json")
    if not os.path.exists(map_path):
        return None
    with open(map_path, encoding="utf-8") as f:
        return json.load(f)


def assign_labels_to_words(char_map, label_spans):
    """char_map: {"char_map": [{word_index, text, bbox, char_start, char_end}, ...]}
    label_spans: [{"start": int, "end": int, "labels": [str]}, ...]

    返回: [{"text": str, "bbox": [...], "label": str}, ...]
    """
    words_out = []

    for word_entry in char_map["char_map"]:
        w_start = word_entry["char_start"]
        w_end = word_entry["char_end"]

        assigned_label = "O"
        for span in label_spans:
            if spans_overlap(w_start, w_end, span["start"], span["end"]):
                # 一个词理论上只应该落在一个标注span里；如果重叠多个，
                # 取第一个匹配到的（正常标注不应出现这种情况）
                assigned_label = span["labels"][0]
                break

        words_out.append({
            "text": word_entry["text"],
            "bbox": word_entry["bbox"],
            "label": assigned_label,
        })

    return words_out


def main():
    with open(ANNOTATIONS_PATH, encoding="utf-8") as f:
        tasks = json.load(f)

    print(f"共 {len(tasks)} 个标注任务\n")

    training_data = []

    for task in tasks:
        source_image = task["data"]["source_image"]
        annotations = task.get("annotations", [])

        if not annotations:
            print(f"跳过（没有标注）: {source_image}")
            continue

        # 取第一个标注者的结果（通常只有一个人标注）
        result = annotations[0]["result"]
        label_spans = [
            {
                "start": r["value"]["start"],
                "end": r["value"]["end"],
                "labels": r["value"]["labels"],
            }
            for r in result
            if r["type"] == "labels" and r["value"]["labels"][0] not in EXCLUDED_FIELDS
        ]

        char_map = load_char_map_for_image(source_image)
        if char_map is None:
            print(f"警告：找不到对应的char_map: {source_image}")
            continue

        words_with_labels = assign_labels_to_words(char_map, label_spans)

        labeled_count = sum(1 for w in words_with_labels if w["label"] != "O")
        print(f"{source_image}: {len(words_with_labels)} 个词，其中 {labeled_count} 个被标注了字段")

        training_data.append({
            "source_image": source_image,
            "words": words_with_labels,
        })

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(training_data, f, ensure_ascii=False, indent=2)

    print(f"\n已保存: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()