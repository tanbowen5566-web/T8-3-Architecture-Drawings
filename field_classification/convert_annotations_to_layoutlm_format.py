"""
convert_annotations_to_layoutlm_format.py

Purpose: Take the field classification annotations exported from Label
         Studio (character ranges + labels), map them back to each word
         (via char_map), and produce the format needed for LayoutLMv3
         training: for each image -> a list of words, each with
         text / bbox / label

The annotations are character-level (start/end based on the concatenated
text); char_map records the character range of each word within the
concatenated text. Here we assign labels to words using an "overlap" check:
if a word's character range overlaps with an annotation span, that span's
label is assigned to the word. Words not covered by any span are labeled
"O" (standard NER convention, meaning "does not belong to any target
field").

Usage:
    python convert_annotations_to_layoutlm_format.py

Input:
    data/field_annotations/project-2-at-2026-08-29-12-04-a62de1c8.json
    data/ocr_words_char_map/*.json

Output:
    data/layoutlm_training_data.json
"""

import json
import os

ANNOTATIONS_PATH = "data/field_annotations/project-2-at-2026-08-29-12-04-a62de1c8.json"
CHAR_MAP_DIR = "data/ocr_words_char_map"
OUTPUT_PATH = "data/layoutlm_training_data.json"

# Fields temporarily excluded: during annotation, drawing_name and
# drawing_types ended up labeled over the same piece of text (overlapping
# spans), but LayoutLMv3 token classification requires each word to have
# only one label. The boundary between the two field definitions hasn't
# been confirmed with the team yet, so skip them for now to get the
# pipeline running end-to-end.
# Once the field definitions are confirmed, clear or adjust the line below
# to bring them back in.
EXCLUDED_FIELDS = {"drawing_name", "drawing_types"}


def spans_overlap(a_start, a_end, b_start, b_end):
    return a_start < b_end and b_start < a_end


def load_char_map_for_image(source_image):
    """char_map filename convention: <image_stem>_char_map.json"""
    stem = os.path.splitext(source_image)[0]
    map_path = os.path.join(CHAR_MAP_DIR, f"{stem}_char_map.json")
    if not os.path.exists(map_path):
        return None
    with open(map_path, encoding="utf-8") as f:
        return json.load(f)


def assign_labels_to_words(char_map, label_spans):
    """char_map: {"char_map": [{word_index, text, bbox, char_start, char_end}, ...]}
    label_spans: [{"start": int, "end": int, "labels": [str]}, ...]

    Returns: [{"text": str, "bbox": [...], "label": str}, ...]
    """
    words_out = []

    for word_entry in char_map["char_map"]:
        w_start = word_entry["char_start"]
        w_end = word_entry["char_end"]

        assigned_label = "O"
        for span in label_spans:
            if spans_overlap(w_start, w_end, span["start"], span["end"]):
                # In theory a word should only fall inside one annotation
                # span; if it overlaps multiple, take the first match
                # (shouldn't happen with correct annotations)
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

    print(f"{len(tasks)} annotation tasks total\n")

    training_data = []

    for task in tasks:
        source_image = task["data"]["source_image"]
        annotations = task.get("annotations", [])

        if not annotations:
            print(f"Skipped (no annotations): {source_image}")
            continue

        # Take the first annotator's result (usually only one person annotates)
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
            print(f"Warning: no matching char_map found: {source_image}")
            continue

        words_with_labels = assign_labels_to_words(char_map, label_spans)

        labeled_count = sum(1 for w in words_with_labels if w["label"] != "O")
        print(f"{source_image}: {len(words_with_labels)} words, {labeled_count} of which were labeled with a field")

        training_data.append({
            "source_image": source_image,
            "words": words_with_labels,
        })

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(training_data, f, ensure_ascii=False, indent=2)

    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()