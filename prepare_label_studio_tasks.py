"""
prepare_label_studio_tasks.py

用途：把 data/ocr_words/*.json（docTR提取的词+坐标）转换成
      Label Studio可导入的任务格式（文本标注），供字段分类标注使用。

做法：
  - 把每张图的词按顺序用空格拼成一段文字（Text组件需要一整段文字）
  - 同时记录每个词对应的字符起止位置（char_start/char_end），
    存一份映射表，方便标注完成后把"字符范围标签"对应回"具体的词+坐标"
    （因为最终训练LayoutLMv3需要的是"每个词的字段标签"，不是字符范围）

用法：
    python prepare_label_studio_tasks.py

输出：
    output/label_studio_import_tasks.json      -> 直接导入Label Studio用
    data/ocr_words_char_map/*.json             -> 词<->字符位置映射（后处理用）
"""

import json
import os

INPUT_DIR = "data/ocr_words"
CHAR_MAP_DIR = "data/ocr_words_char_map"
TASKS_OUTPUT_PATH = "output/label_studio_import_tasks.json"


def build_text_and_char_map(words):
    """把词列表拼成一段文字，同时记录每个词的字符起止位置。
    词与词之间用单个空格分隔。
    """
    text_parts = []
    char_map = []  # 每项: {word_index, text, bbox, char_start, char_end}
    cursor = 0

    for i, w in enumerate(words):
        word_text = w["text"]
        char_start = cursor
        char_end = cursor + len(word_text)

        text_parts.append(word_text)
        char_map.append({
            "word_index": i,
            "text": word_text,
            "bbox": w["bbox"],
            "confidence": w.get("confidence"),
            "char_start": char_start,
            "char_end": char_end,
        })

        cursor = char_end + 1  # +1 for the space we'll join with

    full_text = " ".join(text_parts)
    return full_text, char_map


def main():
    os.makedirs(CHAR_MAP_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(TASKS_OUTPUT_PATH), exist_ok=True)

    json_files = sorted(f for f in os.listdir(INPUT_DIR) if f.endswith(".json"))

    if not json_files:
        print(f"警告：{INPUT_DIR} 里没有找到json文件")
        return

    tasks = []

    for fname in json_files:
        with open(os.path.join(INPUT_DIR, fname), encoding="utf-8") as f:
            data = json.load(f)

        words = data["words"]
        source_image = data["source_image"]

        full_text, char_map = build_text_and_char_map(words)

        # 保存字符映射表，文件名和图片对应
        map_out_name = os.path.splitext(fname)[0] + "_char_map.json"
        map_out_path = os.path.join(CHAR_MAP_DIR, map_out_name)
        with open(map_out_path, "w", encoding="utf-8") as f:
            json.dump({"source_image": source_image, "char_map": char_map}, f, ensure_ascii=False, indent=2)

        # 构建Label Studio任务
        tasks.append({
            "data": {
                "text": full_text,
                "source_image": source_image,
            }
        })

        print(f"{fname}: {len(words)} 个词 -> 文本长度 {len(full_text)} 字符")

    with open(TASKS_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(tasks, f, ensure_ascii=False, indent=2)

    print(f"\n共生成 {len(tasks)} 个任务")
    print(f"导入文件: {TASKS_OUTPUT_PATH}")
    print(f"字符映射表保存在: {CHAR_MAP_DIR}/")


if __name__ == "__main__":
    main()