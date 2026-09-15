"""
keyword_drawing_types.py

Purpose: First-pass drawing_types classification -- map the drawing_name
         text (extracted by the field classification model) to the
         client's fixed vocabulary via keyword matching:

             Elevation | Perspective | Plan | Section | Detail

         (per client spec: https://en.wikipedia.org/wiki/Architectural_drawing)

This only handles the "easy" case where the drawing's title already
states its type (e.g. "BASEMENT PLAN", "EAST ELEVATIONS STAGE I").
The client explicitly flagged that OLDER drawings often do NOT state
the type in the title -- that's the harder case this alone can't solve,
and needs a visual/content-based classifier (CLIP / TU Darmstadt CNN /
similar) as a second stage. Anything that doesn't match a keyword here
(no match, or matches something outside the 5-category vocabulary, e.g.
a "SCHEDULE" title) is left unclassified for that next stage to handle,
not force-fit into the wrong category.

Usage as a library:
    from keyword_drawing_types import classify_drawing_type_from_text
    label = classify_drawing_type_from_text("BASEMENT PLAN STAGE I.")
    # -> "Plan"

Usage as a script (quick spot-check against a batch of drawing_name text):
    python keyword_drawing_types.py --annotations data/project-7-at-2026-09-13-01-26-ff72d5ff.json
"""

import argparse
import json

# Client's fixed vocabulary. Order matters: a title can contain more than
# one keyword (e.g. "ELEVATIONS - SECTION STRUCTURAL ROOF..."), so this is
# checked in order and the FIRST match wins -- earlier entries take
# priority over later ones. Adjust the order if particular combinations
# should resolve differently.
KEYWORD_TO_LABEL = [
    ("PERSPECTIVE", "Perspective"),
    ("ELEVATION", "Elevation"),
    ("SECTION", "Section"),
    ("DETAIL", "Detail"),
    ("PLAN", "Plan"),
]


def classify_drawing_type_from_text(drawing_name_text):
    """Returns one of the 5 official labels, or None if no keyword in the
    text matches the client's vocabulary (e.g. text is blank, is just a
    drawing number, or names something outside the 5 categories like a
    schedule). None means: hand this off to the next stage (visual
    classification or manual review), don't guess."""
    if not drawing_name_text:
        return None

    text_upper = drawing_name_text.upper()
    for keyword, label in KEYWORD_TO_LABEL:
        if keyword in text_upper:
            return label

    return None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Spot-check keyword-based drawing_types classification against human-annotated drawing_name text"
    )
    parser.add_argument("--annotations", required=True, help="Label Studio field classification export (JSON)")
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.annotations, encoding="utf-8") as f:
        tasks = json.load(f)

    matched_count = 0
    unmatched_count = 0
    unmatched_examples = []

    for task in tasks:
        source_image = task["data"]["source_image"]
        annotations = task.get("annotations", [])
        if not annotations:
            continue

        name_text = None
        for r in annotations[0]["result"]:
            if r["type"] == "labels" and "drawing_name" in r["value"]["labels"]:
                name_text = r["value"]["text"]
                break

        if name_text is None:
            continue

        label = classify_drawing_type_from_text(name_text)
        if label is not None:
            matched_count += 1
            print(f"{source_image}")
            print(f"  drawing_name: {name_text!r}  ->  {label}\n")
        else:
            unmatched_count += 1
            unmatched_examples.append((source_image, name_text))

    print("===== Unmatched (need visual classification / manual review) =====")
    for source_image, name_text in unmatched_examples:
        print(f"{source_image}: {name_text!r}")

    total = matched_count + unmatched_count
    print(f"\n===== Summary =====")
    print(f"Matched by keyword: {matched_count}/{total}")
    print(f"Unmatched (need next stage): {unmatched_count}/{total}")


if __name__ == "__main__":
    main()