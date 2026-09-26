"""
evaluate_tu_darmstadt_model.py

Purpose: Evaluate the TU Darmstadt DrawingClassifiers CNN
         (https://github.com/Paulinos739/DrawingClassifiers) against a
         larger, real sample -- using the keyword-derived drawing_types
         (from drawing_name, already validated) as ground truth.

Scope: only Plan / Elevation / Section are evaluated. TU's model only
outputs elevation/floor-plan/section (floor-plan maps to the client's
"Plan" category); Perspective and Detail aren't in its vocabulary at
all, so images with those ground-truth labels are skipped rather than
forced into a comparison that can't be fair.

IMPORTANT preprocessing note: the model's own inference script
(src/PredictDrawingCategory.py in the TU repo) does NOT rescale pixel
values to 0-1 before prediction. Testing without that rescaling
produced degenerate output (predicting the exact same class with 100%
confidence for every image, including random noise) -- almost
certainly because the model was trained with normalized (0-1) input.
This script rescales (divides by 255) before prediction; if you get
similarly degenerate results, that's the signal something is still off
with preprocessing, not that this particular test case is unusual.

Setup (one-time):
    git clone https://github.com/Paulinos739/DrawingClassifiers.git <somewhere outside your git repo>
    pip install tensorflow --break-system-packages

Usage:
    python evaluate_tu_darmstadt_model.py \
        --annotations data/project-7-at-2026-09-13-01-26-ff72d5ff.json \
        --tu-repo-dir /path/to/cloned/DrawingClassifiers \
        --image-search-dirs data/YFA_cabana data/YFA_state_government
"""

import argparse
import json
import os

import numpy as np


# Client vocabulary -> keyword matching (same as keyword_drawing_types.py)
DRAWING_TYPE_KEYWORDS = [
    ("PERSPECTIVE", "Perspective"),
    ("ELEVATION", "Elevation"),
    ("SECTION", "Section"),
    ("DETAIL", "Detail"),
    ("PLAN", "Plan"),
]

# TU model's output category -> client vocabulary
TU_LABEL_TO_CLIENT_LABEL = {
    "elevation": "Elevation",
    "floor-plan": "Plan",
    "section": "Section",
}

# Only these client categories can be fairly evaluated against TU's model
EVALUABLE_CATEGORIES = {"Plan", "Elevation", "Section"}


def classify_drawing_type_from_text(drawing_name_text):
    if not drawing_name_text:
        return None
    text_upper = drawing_name_text.upper()
    for keyword, label in DRAWING_TYPE_KEYWORDS:
        if keyword in text_upper:
            return label
    return None


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate TU Darmstadt drawing classifier against keyword-derived ground truth")
    parser.add_argument("--annotations", required=True, help="Label Studio field classification export (JSON)")
    parser.add_argument("--tu-repo-dir", required=True, help="Path to the cloned DrawingClassifiers repo")
    parser.add_argument("--image-search-dirs", nargs="+", required=True, help="Folders to search for the raw drawing images")
    return parser.parse_args()


def find_image_path(search_dirs, filename):
    for base_dir in search_dirs:
        for root, _dirs, files in os.walk(base_dir):
            if filename in files:
                return os.path.join(root, filename)
    return None


def build_ground_truth(annotations_path):
    """Returns [(filename, ground_truth_label), ...] for images whose
    keyword-derived label is one of Plan/Elevation/Section (the only
    categories TU's model can be fairly compared against)."""
    with open(annotations_path, encoding="utf-8") as f:
        tasks = json.load(f)

    result = []
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
        if label in EVALUABLE_CATEGORIES:
            result.append((source_image, label))

    return result


def main():
    args = parse_args()

    # Import TensorFlow / Keras lazily and load the model from the cloned repo
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing import image as keras_image

    model_path = os.path.join(
        args.tu_repo_dir, "TrainedModels", "DrawingCategoryModels", "classifier_multiclass_drawing_category.hdf5"
    )
    print(f"Loading TU Darmstadt model from: {model_path}")
    model = load_model(model_path, compile=False)
    tu_columns = ["elevation", "floor-plan", "section"]

    print("Building ground truth from keyword-matched drawing_name text...")
    ground_truth = build_ground_truth(args.annotations)
    print(f"{len(ground_truth)} images have an evaluable ground-truth label (Plan/Elevation/Section)\n")

    if not ground_truth:
        print("Nothing to evaluate -- no images with a Plan/Elevation/Section keyword match were found")
        return

    correct = 0
    total = 0
    confusion = {}  # (true_label, predicted_label) -> count
    per_class_total = {}
    per_class_correct = {}
    unresolved_images = []

    for filename, true_label in ground_truth:
        image_path = find_image_path(args.image_search_dirs, filename)
        if image_path is None:
            unresolved_images.append(filename)
            continue

        img = keras_image.load_img(image_path, target_size=(64, 64))
        img_arr = keras_image.img_to_array(img) / 255.0  # rescale -- see module docstring
        img_arr = np.expand_dims(img_arr, axis=0)
        pred = model.predict(img_arr, verbose=0)[0]
        predicted_label = TU_LABEL_TO_CLIENT_LABEL[tu_columns[int(np.argmax(pred))]]

        total += 1
        per_class_total[true_label] = per_class_total.get(true_label, 0) + 1
        if predicted_label == true_label:
            correct += 1
            per_class_correct[true_label] = per_class_correct.get(true_label, 0) + 1

        confusion[(true_label, predicted_label)] = confusion.get((true_label, predicted_label), 0) + 1

        match = "OK " if predicted_label == true_label else "ERR"
        print(f"{match} {filename}: true={true_label:10s} predicted={predicted_label:10s} scores={[round(float(s), 2) for s in pred]}")

    if unresolved_images:
        print(f"\nWarning: {len(unresolved_images)} image(s) not found in the search directories, skipped:")
        for fname in unresolved_images:
            print(f"   {fname}")

    print(f"\n===== Overall =====")
    print(f"Accuracy: {correct}/{total} = {correct/total:.1%}" if total else "No images evaluated")

    print("\nPer-class recall:")
    for label in sorted(per_class_total.keys()):
        t = per_class_total[label]
        c = per_class_correct.get(label, 0)
        print(f"  {label:10s} {c}/{t} = {c/t:.1%}")

    print("\nConfusion matrix (true -> predicted : count):")
    for (true_label, pred_label), count in sorted(confusion.items()):
        print(f"  {true_label:10s} -> {pred_label:10s} : {count}")


if __name__ == "__main__":
    main()