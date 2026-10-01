"""
run_pipeline.py

Purpose: End-to-end entry point that chains together the two stages of
         the pipeline into a single command:
           1. Title block detection + crop (Faster R-CNN, v2 model)
           2. OCR + field classification -> structured CSV (docTR + LayoutLMv3)


2026-09-14 update: field classification now uses field_classifier_v1
(84-image training set, early-stopped at the best validation-loss
checkpoint) instead of field_classifier_v0 (5-image proof of concept).
On the held-out validation split, v1 gets 100% recall on architect,
96% on date, 91% on project_name; draughtsperson/drawing_number/scale
are weaker (62-70%), partly due to a handful of images with incomplete
annotations rather than purely model error -- see team notes.

2026-09-14 update (2): drawing_types is now derived from the
drawing_name text via keyword matching against the client's fixed
vocabulary (Elevation / Perspective / Plan / Section / Detail --
see https://en.wikipedia.org/wiki/Architectural_drawing). This only
covers drawings whose title already states the type (~78% of the
84-image annotation batch) -- the client explicitly noted that older
drawings often do NOT state the type in the title, which keyword
matching can't help with at all (nothing to match against). Those
cases, plus titles matching a keyword outside the 5-category vocabulary
(e.g. "SCHEDULE"), are left unclassified here (drawing_types blank,
needs_manual_review = True) pending a visual/content-based classifier
(CLIP was tried and discriminated poorly on these hand-drafted line
drawings; TU Darmstadt's DrawingClassifiers project -- 3 of 5
categories -- is a candidate but not yet verified runnable). See
drawing_type_classification/ for the exploration scripts this was
developed from, including the CLIP attempt.

This also switches to field_classifier_v2, which re-includes
drawing_name in training (previously excluded alongside drawing_types
due to the same annotation-span conflict -- see
convert_annotations_to_layoutlm_format.py's notes for why only
drawing_types needs to stay excluded now). On the held-out validation
split, v2 gets 88% recall on drawing_name, and every other field's
recall also improved over v1 (e.g. scale 70% -> 90%).

2026-09-26 update: No-TB routing. Drawings where the detector finds no
high-confidence title block (score < DETECTION_SCORE_THRESHOLD) used to be
skipped entirely (only logged in *_failed.txt). They now go through the
No-TB branch (src/no_tb/branch.py): full image -> docTR -> the same
field_classifier_v2, with a sliding window over tokens and one spatial span
per field. This is a BASELINE -- the classifier was trained on title block
crops only -- so every No-TB row is marked needs_manual_review = True.
Stage 2 now iterates over this run's routing result instead of listing the
crop folder, so stale crops from an earlier run with the same run_name are
no longer picked up. Use --no_tb skip to restore the old behaviour.

2026-09-27 update: the default No-TB method is now the rule/layout-based
extractor (--no_tb_method rules, src/no_tb/rules.py): preprocessing, tiled
docTR, ink colour, text blocks, block roles (title / view_title / scale /
drawing_no / stamp / contract_note / archive_mark ...), then fields. The
LayoutLMv3 full-sheet baseline is still available with --no_tb_method layoutlm.

Usage:
    # Dataset mode -- uses the team's iter_images(dataset_name), which
    # reads data/<dataset_name>/metadata.csv and each Node ID's folder.
    # This is the mode that gives Node ID traceability in the output.
    python run_pipeline.py --dataset_name YFA_cabana --run_name yfa_cabana

    # Quick demo run on just the first N images of a dataset
    python run_pipeline.py --dataset_name YFA_cabana --run_name demo --limit 10

    # Folder mode -- arbitrary folder, no metadata.csv required.
    # Traceability is by filename only (no Node ID available).
    python run_pipeline.py --input_dir "/path/to/SomeFolder" --run_name some_batch

    # Multiple input folders at once are also supported in folder mode
    python run_pipeline.py --input_dir folder1 folder2 --run_name some_batch

    # Run with no arguments to use the default datasets and run name (folder mode)
    python run_pipeline.py

    # No-TB branch options
    python run_pipeline.py --input_dir folder --run_name x --no_tb skip          # old behaviour
    python run_pipeline.py --input_dir folder --run_name x --no_tb_debug                  # rules + overlays
    python run_pipeline.py --input_dir folder --run_name x --no_tb_method layoutlm       # old baseline
    python run_pipeline.py --input_dir folder --run_name x --no_tb_method layoutlm --no_tb_ocr tiled --no_tb_upscale 2

2026-09-27 update: single output file. Values and confidences are now written
to ONE csv instead of the old main csv + *_confidence.csv pair:
    source_image, node_id, <field>, <field>_confidence, ... (OUTPUT_FIELDS order),
    then run info (EXTRA_COLUMNS: branch, tb_score, drawing_types_source,
    needs_manual_review). Written with save_output(..., columns=...), which
    defaults to the 8-column schema for other callers.

Folder layout:
    models/        trained models the pipeline LOADS (read-only at run time)
    intermediate/  files produced DURING a run, one sub-folder per run_name
    results/       the final output of a run
(Models still found in the old output/ location are used with a warning.)

This creates:
    intermediate/<run_name>/titleblock_crops/     cropped title block images
    intermediate/<run_name>/routing.csv           per image: branch, detector score, crop path
    intermediate/<run_name>/no_tb_images.txt      images without a detected title block
    intermediate/<run_name>/no_tb_debug/          only with --no_tb_debug: per-drawing json + overlay
    results/extracted_metadata_<run_name>.csv     THE output: source_image, node_id, then each
                                                  field followed by its confidence
                                                  (node_id is only populated in dataset
                                                  mode; blank in folder mode)
"""

import argparse
import csv
import os

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification

from src.input_output import save_output, iter_images
from src.no_tb import process_no_tb

# ---------------------------------------------------------------------------
# Folder layout
# ---------------------------------------------------------------------------
MODELS_DIR = "models"              # trained models the pipeline loads
INTERMEDIATE_DIR = "intermediate"  # files produced during a run (per run_name)
RESULTS_DIR = "results"            # final output of a run
LEGACY_MODELS_DIR = "output"       # old location, still accepted with a warning

DETECTOR_MODEL_PATH = os.path.join(MODELS_DIR, "title_block_detector_v2.pth")
FIELD_MODEL_DIR = os.path.join(MODELS_DIR, "field_classifier_v2")


def resolve_model_path(path):
    """Model path under models/; falls back to the old output/ location so
    older checkouts / download scripts keep working until they are moved."""
    if os.path.exists(path):
        return path
    legacy = os.path.join(LEGACY_MODELS_DIR, os.path.basename(path))
    if os.path.exists(legacy):
        print(f"Warning: using model from old location '{legacy}'. "
              f"Move it to '{path}'.")
        return legacy
    raise FileNotFoundError(
        f"Model not found: '{path}' (also checked '{legacy}'). "
        f"Put the trained model in {MODELS_DIR}/ or run download_models.py."
    )
DEFAULT_SOURCE_DIRS = [
    "data/YFA_cabana",
    "data/YFA_cabana_test",
    "data/YFA_state_government",
]

DEFAULT_RUN_NAME = "v2"

DETECTOR_NUM_CLASSES = 2
DETECTION_SCORE_THRESHOLD = 0.5  # Only trust high-confidence boxes for auto-cropping
CROP_PADDING = 20

# drawing_name is now included so drawing_types can be derived from it
# below; drawing_types itself is never predicted by the LayoutLMv3 model.
MODEL_FIELDS = [
    "architect",
    "project_name",
    "draughtsperson",
    "date",
    "scale",
    "drawing_number",
    "drawing_name",
]

# Output column order: source_image, node_id, then each field immediately
# followed by its confidence, then EXTRA_COLUMNS. Order of the fields
# follows the team's 8-column schema.
OUTPUT_FIELDS = [
    "project_name",
    "drawing_name",
    "architect",
    "draughtsperson",
    "drawing_number",
    "date",
    "scale",
    "drawing_types",
]
# Run info appended after the field columns. Set to [] for a pure
# source_image + node_id + fields/confidences file.
EXTRA_COLUMNS = ["branch", "tb_score", "drawing_types_source", "needs_manual_review"]


def output_fieldnames():
    cols = ["source_image", "node_id"]
    for f in OUTPUT_FIELDS:
        cols += [f, f"{f}_confidence"]
    return cols + EXTRA_COLUMNS


def fmt_conf(conf):
    return f"{conf:.3f}" if conf is not None else ""


# Client's fixed drawing_types vocabulary. Order matters: a title can
# contain more than one keyword (e.g. "ELEVATIONS - SECTION STRUCTURAL
# ROOF..."), so this is checked in order and the FIRST match wins.
# See drawing_type_classification/keyword_drawing_types.py for the
# standalone version this was developed from, including a spot-check
# script against the annotation export.
DRAWING_TYPE_KEYWORDS = [
    ("PERSPECTIVE", "Perspective"),
    ("ELEVATION", "Elevation"),
    ("SECTION", "Section"),
    ("DETAIL", "Detail"),
    ("PLAN", "Plan"),
]


def classify_drawing_type_from_text(drawing_name_text):
    """Returns one of the 5 official labels, or None if no keyword in the
    text matches the client's vocabulary (blank text, a bare drawing
    number, or something outside the 5 categories like a schedule).
    None means: needs a visual classifier or manual review, don't guess."""
    if not drawing_name_text:
        return None
    text_upper = drawing_name_text.upper()
    for keyword, label in DRAWING_TYPE_KEYWORDS:
        if keyword in text_upper:
            return label
    return None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the full pipeline: detect + crop title blocks, then OCR + classify fields into a CSV"
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--dataset_name",
        help="Name of a team dataset with a metadata.csv (e.g. YFA_cabana). Uses iter_images() so the output can be traced back to Node ID. Mutually exclusive with --input_dir.",
    )
    mode_group.add_argument(
        "--input_dir", nargs="+", default=None,
        help="One or more arbitrary folders of raw images (project-relative or absolute path, e.g. a folder on the Desktop). No metadata.csv required, but the output cannot be traced to a Node ID -- only by filename. Mutually exclusive with --dataset_name.",
    )
    parser.add_argument(
        "--run_name", default=DEFAULT_RUN_NAME,
                help="Short name used to build all paths for this run (e.g. 'run_name=demo' produces intermediate/demo/ and results/extracted_metadata_demo.csv)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Only process the first N images (useful for a quick demo run instead of the full dataset)",
    )
    parser.add_argument(
        "--no_tb", choices=["tb_recipe", "skip"], default="tb_recipe",
        help="What to do with drawings where no title block is detected: 'tb_recipe' = run the TB field classifier on the full sheet (src/no_tb), 'skip' = old behaviour (logged only)",
    )
    parser.add_argument(
        "--no_tb_method", choices=["rules", "layoutlm"], default="rules",
        help="No-TB extractor: 'rules' = rule/layout-based extractor (src/no_tb/rules.py, default), 'layoutlm' = baseline, the TB field classifier on the full sheet (src/no_tb/baseline.py)",
    )
    parser.add_argument(
        "--no_tb_target_long", type=int, default=4096,
        help="No-TB rules only: long side (px) the sheet is resized to before tiled OCR. Lower (e.g. 3072) = faster, may miss small lettering",
    )
    parser.add_argument(
        "--no_tb_ocr", choices=["full", "tiled"], default="full",
        help="No-TB layoutlm baseline only. OCR mode: 'full' = one docTR pass on the whole sheet (same as TB branch), 'tiled' = 1024px overlapping tiles so small lettering is not downscaled away",
    )
    parser.add_argument(
        "--no_tb_debug", action="store_true",
        help="No-TB only: dump every OCR word + predicted label (json) and an overlay image to intermediate/<run_name>/no_tb_debug/",
    )
    parser.add_argument(
        "--no_tb_upscale", type=float, default=1.0,
        help="No-TB layoutlm baseline only: resize factor applied to the sheet before OCR (e.g. 2 for 2048px-wide scans)",
    )
    args = parser.parse_args()
    if args.dataset_name is None and args.input_dir is None:
        args.input_dir = DEFAULT_SOURCE_DIRS
    return args


# ---------------------------------------------------------------------------
# Image discovery: either a plain folder scan, or the team's dataset format
# (metadata.csv + Node ID folders) via iter_images(), which additionally
# gives us Node ID traceability.
# ---------------------------------------------------------------------------

def find_all_images(source_dirs):
    """Scan all source directories and return a list of
    (real filename, full path). Duplicate filenames are only kept once
    (e.g. files that exist in both YFA_cabana_test and YFA_cabana)."""
    seen_names = set()
    result = []
    for base_dir in source_dirs:
        if not os.path.isdir(base_dir):
            print(f"Warning: directory not found, skipping: {base_dir}")
            continue
        for root, _dirs, files in os.walk(base_dir):
            for fname in sorted(files):
                if not fname.lower().endswith(".jpg"):
                    continue
                if fname in seen_names:
                    continue
                seen_names.add(fname)
                result.append((fname, os.path.join(root, fname)))
    return result


def get_images_and_node_ids(args):
    """Returns (images, node_id_by_filename).
    images is a list of (filename, full_path).
    node_id_by_filename maps filename -> Node ID string, and is only
    populated in dataset mode; empty dict in folder mode."""
    if args.dataset_name:
        print(f"Dataset mode: reading '{args.dataset_name}' via iter_images() (Node ID tracking enabled)\n")
        records = list(iter_images(args.dataset_name))
        images = [(r["image_name"], str(r["image_path"])) for r in records]
        node_id_by_filename = {r["image_name"]: r["node_id"] for r in records}
        return images, node_id_by_filename
    else:
        print("Folder mode: scanning provided folders (no Node ID tracking -- filename only)\n")
        images = find_all_images(args.input_dir)
        return images, {}


# ---------------------------------------------------------------------------
# Stage 1: detection + crop
# ---------------------------------------------------------------------------

def build_detector(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def detect_and_crop_all(images, crop_output_dir, failed_log_path, routing_csv_path, device):
    """Detect + crop, and route every image to a branch.
    Returns a list (same order as images) of
      {"fname", "image_path", "branch": "tb" | "no_tb", "tb_score", "crop_path"}"""
    print("Loading title block detection model (v2)...")
    detector = build_detector(DETECTOR_NUM_CLASSES)
    detector.load_state_dict(torch.load(resolve_model_path(DETECTOR_MODEL_PATH), map_location=device))
    detector.to(device)
    detector.eval()

    os.makedirs(crop_output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(failed_log_path) or ".", exist_ok=True)

    print(f"Found {len(images)} images to process\n")

    success_count = 0
    failed_images = []
    routes = []

    for i, (fname, img_path) in enumerate(images, 1):
        image = Image.open(img_path).convert("RGB")
        img_tensor = torchvision.transforms.functional.to_tensor(image).to(device)

        with torch.no_grad():
            prediction = detector([img_tensor])[0]

        boxes = prediction["boxes"]
        scores = prediction["scores"]

        max_score = scores.max().item() if len(scores) else 0.0
        if len(scores) == 0 or max_score < DETECTION_SCORE_THRESHOLD:
            print(f"[{i}/{len(images)}] No title block (best score={max_score:.3f}) -> No-TB branch: {fname}")
            failed_images.append(fname)
            routes.append({"fname": fname, "image_path": img_path, "branch": "no_tb",
                           "tb_score": max_score, "crop_path": ""})
            continue

        best_idx = scores.argmax()
        x1, y1, x2, y2 = boxes[best_idx].tolist()
        score = scores[best_idx].item()

        x1 = max(0, x1 - CROP_PADDING)
        y1 = max(0, y1 - CROP_PADDING)
        x2 = min(image.width, x2 + CROP_PADDING)
        y2 = min(image.height, y2 + CROP_PADDING)

        cropped = image.crop((x1, y1, x2, y2))
        out_path = os.path.join(crop_output_dir, fname)
        cropped.save(out_path)

        print(f"[{i}/{len(images)}] Success (confidence={score:.3f}): {fname}")
        success_count += 1
        routes.append({"fname": fname, "image_path": img_path, "branch": "tb",
                       "tb_score": score, "crop_path": out_path})

    with open(failed_log_path, "w", encoding="utf-8") as f:
        for fname in failed_images:
            f.write(fname + "\n")

    with open(routing_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["fname", "image_path", "branch", "tb_score", "crop_path"])
        writer.writeheader()
        for r in routes:
            writer.writerow({**r, "tb_score": f"{r['tb_score']:.3f}"})

    print(f"\nDetection done: {success_count} with title block, {len(failed_images)} without (-> No-TB branch)")
    print(f"Crops saved to: {crop_output_dir}")
    print(f"No-TB list saved to: {failed_log_path}")
    print(f"Routing saved to: {routing_csv_path}\n")
    return routes


# ---------------------------------------------------------------------------
# Stage 2: OCR + field classification -> CSV
# ---------------------------------------------------------------------------

def normalize_bbox_0_1_to_1000(bbox):
    x0, y0, x1, y1 = bbox
    return [
        max(0, min(1000, int(x0 * 1000))),
        max(0, min(1000, int(y0 * 1000))),
        max(0, min(1000, int(x1 * 1000))),
        max(0, min(1000, int(y1 * 1000))),
    ]


def extract_words_with_boxes(image_path, ocr_model):
    doc = DocumentFile.from_images(str(image_path))
    result = ocr_model(doc)

    items = []
    for page in result.pages:
        for block in page.blocks:
            for line in block.lines:
                for word in line.words:
                    (x0, y0), (x1, y1) = word.geometry
                    items.append({"text": word.value, "bbox": [x0, y0, x1, y1]})
    return items


def predict_fields(image, words_data, processor, field_model):
    """Run field classification on one image's OCR word list. Returns:
    { field_name: {"value": "joined text", "confidence": average confidence} }
    Fields with no words assigned get value="" and confidence=None.
    """
    id2label = field_model.config.id2label

    words = [w["text"] for w in words_data]
    boxes = [normalize_bbox_0_1_to_1000(w["bbox"]) for w in words_data]

    encoding = processor(
        image, words, boxes=boxes,
        truncation=True, padding="max_length", max_length=512,
        return_tensors="pt",
    )

    with torch.no_grad():
        outputs = field_model(**encoding)

    probs = torch.softmax(outputs.logits, dim=-1).squeeze(0)
    predictions = probs.argmax(-1).tolist()
    word_ids = encoding.word_ids(batch_index=0)

    word_predictions = {}
    for token_idx, word_idx in enumerate(word_ids):
        if word_idx is None or word_idx in word_predictions:
            continue
        label = id2label[predictions[token_idx]]
        confidence = probs[token_idx][predictions[token_idx]].item()
        word_predictions[word_idx] = (label, confidence)

    field_texts = {f: [] for f in MODEL_FIELDS}
    field_confidences = {f: [] for f in MODEL_FIELDS}

    for word_idx, word_text in enumerate(words):
        if word_idx not in word_predictions:
            continue
        label, confidence = word_predictions[word_idx]
        if label in MODEL_FIELDS:
            field_texts[label].append(word_text)
            field_confidences[label].append(confidence)

    result = {}
    for field in MODEL_FIELDS:
        if field_texts[field]:
            value = " ".join(field_texts[field])
            avg_confidence = sum(field_confidences[field]) / len(field_confidences[field])
        else:
            value = ""
            avg_confidence = None
        result[field] = {"value": value, "confidence": avg_confidence}

    return result


def classify_fields_to_csv(routes, output_csv_path, node_id_by_filename, args):
    print("Loading OCR model...")
    ocr_model = ocr_predictor(pretrained=True)

    print("Loading field classification model (v2)...")
    field_model_dir = resolve_model_path(FIELD_MODEL_DIR)
    processor = LayoutLMv3Processor.from_pretrained(field_model_dir, apply_ocr=False)
    field_model = LayoutLMv3ForTokenClassification.from_pretrained(field_model_dir)
    field_model.eval()

    os.makedirs(os.path.dirname(output_csv_path) or ".", exist_ok=True)

    if args.no_tb == "skip":
        skipped = [r for r in routes if r["branch"] == "no_tb"]
        routes = [r for r in routes if r["branch"] == "tb"]
        if skipped:
            print(f"--no_tb skip: {len(skipped)} No-TB image(s) not processed")

    if not routes:
        print("Warning: no images to classify")
        return

    n_tb = sum(r["branch"] == "tb" for r in routes)
    print(f"{len(routes)} images to process ({n_tb} TB, {len(routes) - n_tb} No-TB)\n")

    rows = []
    n_no_tb = 0

    for route in routes:
        fname = route["fname"]
        branch = route["branch"]
        print(f"Processing [{branch}]: {fname}")

        if branch == "tb":
            image_path = route["crop_path"]
            image = Image.open(image_path).convert("RGB")
            words_data = extract_words_with_boxes(image_path, ocr_model)
            if not words_data:
                print("  OCR found no text, skipping")
                continue
            fields = predict_fields(image, words_data, processor, field_model)
        else:
            fields, info = process_no_tb(
                route["image_path"], ocr_model, processor, field_model, MODEL_FIELDS,
                method=args.no_tb_method, target_long=args.no_tb_target_long,
                ocr_mode=args.no_tb_ocr, upscale=args.no_tb_upscale,
                debug_dir=os.path.join(args.work_dir, "no_tb_debug") if args.no_tb_debug else None,
            )
            if fields is None:
                print("  OCR found no text, skipping")
                continue
            print(f"  No-TB [{args.no_tb_method}]: " + ", ".join(f"{k}={v}" for k, v in info.items()))

        # drawing_types: keyword match against the drawing_name text this
        # model just extracted. A match is a deterministic text lookup, not
        # a probabilistic guess, so it gets full confidence (1.0). No match
        # means the title doesn't state a type (or names something outside
        # the 5-category vocabulary, e.g. a schedule) -- flagged for manual
        # review rather than guessed, since there's no visual classifier
        # wired in yet to attempt these.
        drawing_name_text = fields["drawing_name"]["value"]
        drawing_types_label = classify_drawing_type_from_text(drawing_name_text)
        if drawing_types_label is not None:
            drawing_types_value = drawing_types_label
            drawing_types_confidence = 1.0
            drawing_types_source = "keyword"
            needs_manual_review = False
        else:
            drawing_types_value = ""
            drawing_types_confidence = None
            drawing_types_source = "unmatched"
            needs_manual_review = True

        # No-TB output (rules or layoutlm baseline) is not validated against
        # ground truth yet -- always send it to review until it is.
        if branch == "no_tb":
            needs_manual_review = True

        values = {f: fields[f]["value"] for f in MODEL_FIELDS}
        confs = {f: fields[f]["confidence"] for f in MODEL_FIELDS}
        values["drawing_types"] = drawing_types_value
        confs["drawing_types"] = drawing_types_confidence

        row = {"source_image": fname, "node_id": node_id_by_filename.get(fname, "")}
        for f in OUTPUT_FIELDS:
            row[f] = values.get(f, "")
            row[f"{f}_confidence"] = fmt_conf(confs.get(f))
        extra = {
            "branch": branch,
            "tb_score": f"{route['tb_score']:.3f}",
            "drawing_types_source": drawing_types_source,
            "needs_manual_review": needs_manual_review,
        }
        for c in EXTRA_COLUMNS:
            row[c] = extra[c]
        rows.append(row)
        n_no_tb += branch == "no_tb"

        print("  Done")

    save_output(rows, output_csv_path, columns=output_fieldnames())

    print(f"\nField classification done: {len(rows)} images processed "
          f"({len(rows) - n_no_tb} TB, {n_no_tb} No-TB)")
    if not node_id_by_filename:
        print("Note: running in folder mode -- 'node_id' column is blank. Use --dataset_name for Node ID traceability.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print("===== Stage 0: start =====\n")
    args = parse_args()
    device = torch.device("cpu")

    # everything produced during the run -> intermediate/<run_name>/
    args.work_dir = os.path.join(INTERMEDIATE_DIR, args.run_name)
    crop_output_dir = os.path.join(args.work_dir, "titleblock_crops")
    failed_log_path = os.path.join(args.work_dir, "no_tb_images.txt")
    routing_csv_path = os.path.join(args.work_dir, "routing.csv")
    # final output -> results/
    output_csv_path = os.path.join(RESULTS_DIR, f"extracted_metadata_{args.run_name}.csv")
    os.makedirs(args.work_dir, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    images, node_id_by_filename = get_images_and_node_ids(args)

    if args.limit is not None:
        images = images[:args.limit]
        print(f"--limit {args.limit} applied: processing {len(images)} image(s)\n")

    print("===== Stage 1: title block detection + crop + routing =====\n")
    routes = detect_and_crop_all(images, crop_output_dir, failed_log_path, routing_csv_path, device)

    print("===== Stage 2: OCR + field classification (TB crop / No-TB full sheet) =====\n")
    classify_fields_to_csv(routes, output_csv_path, node_id_by_filename, args)
    
    print(f"\nIntermediate files: {args.work_dir}/")
    print(f"Final result:       {output_csv_path}")


if __name__ == "__main__":
    main()