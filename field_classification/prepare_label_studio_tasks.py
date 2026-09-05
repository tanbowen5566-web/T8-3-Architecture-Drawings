"""
prepare_label_studio_tasks.py

Purpose: Convert data/ocr_words/*.json (words + coordinates extracted by
         docTR) into a task format importable by Label Studio (text
         annotation), for use in field classification annotation.

Approach:
  - Join each image's words in order with spaces into a single block of
    text (the Text component needs one continuous piece of text)
  - At the same time, record the character start/end position
    (char_start/char_end) for each word, saving a mapping table so that
    after annotation is done, the "character range labels" can be mapped
    back to the "specific word + coordinates" (because training LayoutLMv3
    ultimately needs "each word's field label", not character ranges)

Usage:
    python prepare_label_studio_tasks.py

Output:
    output/label_studio_import_tasks.json      -> for direct import into Label Studio
    data/ocr_words_char_map/*.json             -> word <-> character position mapping (for post-processing)
"""

import json
import os

INPUT_DIR = "data/ocr_words_v2"
CHAR_MAP_DIR = "data/ocr_words_char_map_v2"
TASKS_OUTPUT_PATH = "output/label_studio_import_tasks_v2.json"


def build_text_and_char_map(words):
    """Join the word list into a single block of text, while recording each
    word's character start/end position.
    Words are separated by a single space.
    """
    text_parts = []
    char_map = []  # each item: {word_index, text, bbox, char_start, char_end}
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
        print(f"Warning: no json files found in {INPUT_DIR}")
        return

    tasks = []

    for fname in json_files:
        with open(os.path.join(INPUT_DIR, fname), encoding="utf-8") as f:
            data = json.load(f)

        words = data["words"]
        source_image = data["source_image"]

        full_text, char_map = build_text_and_char_map(words)

        # Save the character mapping table, filename corresponds to the image
        map_out_name = os.path.splitext(fname)[0] + "_char_map.json"
        map_out_path = os.path.join(CHAR_MAP_DIR, map_out_name)
        with open(map_out_path, "w", encoding="utf-8") as f:
            json.dump({"source_image": source_image, "char_map": char_map}, f, ensure_ascii=False, indent=2)

        # Build the Label Studio task
        tasks.append({
            "data": {
                "text": full_text,
                "source_image": source_image,
            }
        })

        print(f"{fname}: {len(words)} words -> text length {len(full_text)} characters")

    with open(TASKS_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(tasks, f, ensure_ascii=False, indent=2)

    print(f"\n{len(tasks)} tasks generated")
    print(f"Import file: {TASKS_OUTPUT_PATH}")
    print(f"Character mapping tables saved in: {CHAR_MAP_DIR}/")


if __name__ == "__main__":
    main()