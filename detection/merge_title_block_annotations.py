"""
merge_title_block_annotations.py

Purpose: Merge multiple batches of title block detection annotations
         exported from Label Studio (COCO format), automatically handling:
         1. Deduplication -- the same image may have been annotated multiple
            times across different batches, or multiple times within the same
            batch (dedupe by real filename, not image_id, since image_ids in
            different batches are numbered independently and can't be
            compared directly).
         2. Path resolution -- automatically search the local data/ directory
            for the real path corresponding to each filename, instead of
            manually matching them one by one like before.
         3. Category name correction -- some annotation batches misspelled
            "title_block" as "title_blcok"; this is standardized here
            (this script only uses the bbox, so the category name doesn't
            affect the logic, this note is just for reference).

Usage:
    python merge_title_block_annotations.py

Output:
    data/merged_title_block_annotations.json
    Format: {"image_id_to_path": {...}, "annotations": [...]}
    Each annotation: {"image_id": int, "bbox": [x,y,w,h], "source_batch": str}
"""

import json
import os

# Annotation batches to merge (COCO format, each is a folder containing
# images/ + result.json)
ANNOTATION_BATCHES = [
    "data/project-1-YFA",
    "data/project-1-GOV",
]

# Directories to search for real source image paths (add/remove as needed)
SEARCH_DIRS = [
    "data/YFA_cabana",
    "data/YFA_state_government",
    "data/Bates_smart_Mccutcheon",
]

OUTPUT_PATH = "data/merged_title_block_annotations.json"


def build_filename_to_path_index(search_dirs):
    """Scan the local directories and build an index mapping real filename
    (without path) -> full path"""
    index = {}
    for base_dir in search_dirs:
        if not os.path.isdir(base_dir):
            continue
        for root, _dirs, files in os.walk(base_dir):
            for fname in files:
                if fname.lower().endswith(".jpg"):
                    if fname in index:
                        # Same filename appears in multiple locations (e.g.
                        # duplicated between YFA_cabana_test and YFA_cabana);
                        # keep the first one found, don't overwrite, to avoid
                        # the path flip-flopping
                        continue
                    index[fname] = os.path.join(root, fname)
    return index


def extract_real_filename(label_studio_file_name):
    """Label Studio exports file_name in a form like:
    ..\\..\\label-studio\\media\\upload\\1\\54c8e959-1238196_1984-0047-02635-00001.jpg
    The real filename is the last segment, with the Label Studio uuid prefix
    (uuid-real_filename) stripped off.
    """
    basename = label_studio_file_name.replace("\\", "/").split("/")[-1]
    # The uuid prefix looks like "54c8e959-"; split on the first "-" only,
    # since the real filename itself may also contain "-". The uuid is
    # fixed-length: 8 hex chars + "-".
    if len(basename) > 9 and basename[8] == "-":
        return basename[9:]
    return basename


def main():
    print("Scanning local images, building filename index...")
    filename_index = build_filename_to_path_index(SEARCH_DIRS)
    print(f"Indexed {len(filename_index)} unique filenames\n")

    # Dedupe by real filename: {real_filename: {"bbox": [...], "source_batch": str}}
    merged_by_filename = {}
    skipped_duplicates = []
    unresolved = []

    for batch_dir in ANNOTATION_BATCHES:
        result_path = os.path.join(batch_dir, "result.json")
        if not os.path.exists(result_path):
            print(f"Warning: {result_path} not found, skipping this batch")
            continue

        with open(result_path, encoding="utf-8") as f:
            data = json.load(f)

        # image_id -> real filename
        id_to_realname = {
            img["id"]: extract_real_filename(img["file_name"])
            for img in data["images"]
        }

        # image_id -> bbox (assume each image has only one title_block box, take the first)
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

    print(f"Unique images after merge: {len(merged_by_filename)}")
    if skipped_duplicates:
        print(f"Skipped duplicate annotations: {len(skipped_duplicates)}")
        for name, batch in skipped_duplicates:
            print(f"   {name} (from {batch}, an earlier batch's annotation already kept, keeping the one added first)")

    # Resolve real paths
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
        print(f"\nWarning: {len(unresolved)} files could not be resolved to a local path, skipped:")
        for name in unresolved:
            print(f"   {name}")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {"image_id_to_path": image_id_to_path, "annotations": annotations},
            f, ensure_ascii=False, indent=2,
        )

    print(f"\nFinal number of images usable for training: {len(image_id_to_path)}")
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()