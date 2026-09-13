"""
convert_annotations_to_layoutlm_format.py

Purpose: Convert Label Studio's character-level field classification
         annotations back to word-level, using the char_map generated
         during annotation prep, producing the format needed for
         LayoutLMv3 training: each image -> list of words, each word
         with text / bbox / label.

Annotations are character-level (start/end offsets into the joined
text). char_map records each word's character range within that same
joined text. We assign a label to a word if its character range
overlaps a labeled span. Words with no overlapping span get "O"
(the standard NER "outside" label, meaning "not part of any target
field").

Usage:
    python convert_annotations_to_layoutlm_format.py

Input:
    data/project-7-at-2026-09-13-01-26-ff72d5ff.json   (Label Studio export, 84 tasks)
    data/ocr_words_char_map_v2/*.json

Output:
    data/layoutlm_training_data_v2.json
"""

import json
import os

ANNOTATIONS_PATH = "data/project-7-at-2026-09-13-01-26-ff72d5ff.json"
CHAR_MAP_DIR = "data/ocr_words_char_map_v2"
OUTPUT_PATH = "data/layoutlm_training_data_v2.json"

# Temporarily excluded fields: drawing_name and drawing_types were often
# labelled on the exact same span of text during annotation (overlapping
# spans), but LayoutLMv3's token classification requires each word to have
# exactly one label. The boundary between these two fields still hasn't
# been confirmed with the team, so we skip them for now to keep the
# pipeline running. Clear this set (or adjust it) once the field
# definitions are confirmed and re-annotated cleanly.
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
                # A word should ideally fall inside only one labelled span;
                # if it overlaps multiple, take the first match found
                # (shouldn't normally happen with careful annotation)
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
            print(f"Skipped (no annotation): {source_image}")
            continue

        # Take the first annotator's result (usually only one annotator)
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
            print(f"Warning: no matching char_map found for: {source_image}")
            continue

        words_with_labels = assign_labels_to_words(char_map, label_spans)

        labeled_count = sum(1 for w in words_with_labels if w["label"] != "O")
        print(f"{source_image}: {len(words_with_labels)} words, {labeled_count} labelled with a field")

        training_data.append({
            "source_image": source_image,
            "words": words_with_labels,
        })

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(training_data, f, ensure_ascii=False, indent=2)

    print(f"\nSaved: {OUTPUT_PATH}")
    print(f"Total images with usable annotations: {len(training_data)}")


if __name__ == "__main__":
    main()