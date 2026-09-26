"""
src/no_tb/branch.py

Entry point of the No-TB branch, called by run_pipeline.py for drawings where
the title block detector finds no high-confidence title block.

    method="rules"    (default) rule/layout-based extractor, see rules.py:
                      preprocess -> tiled docTR -> ink colour -> lines/blocks
                      -> block roles -> fields
    method="layoutlm" baseline: the TB branch's LayoutLMv3 on the full sheet,
                      see baseline.py (kept for comparison)

Both return (fields, info) where fields has the same shape as
run_pipeline.predict_fields(): {field: {"value": str, "confidence": float | None}},
so everything downstream (drawing_types, output CSV) is shared with the TB branch.
fields is None if OCR found no text.
"""
import os

from PIL import Image

from .baseline import run_baseline
from .rules import run_rules

Image.MAX_IMAGE_PIXELS = None  # archival scans can be very large


def process_no_tb(image_path, ocr_model, processor, field_model, model_fields,
                  method="rules", ocr_mode="full", upscale=1.0, tile=1024, stride=128,
                  target_long=4096, debug_dir=None):
    """
    method       "rules" | "layoutlm"
    target_long  rules only: long side the sheet is resized to before OCR
    ocr_mode, upscale, stride
                 layoutlm baseline only (same meaning as before)
    debug_dir    if set, writes <name>.json + <name>_overlay.jpg per drawing
    """
    image = Image.open(image_path).convert("RGB")
    name = os.path.basename(str(image_path))

    if method == "layoutlm":
        if upscale != 1.0:
            image = image.resize((int(image.width * upscale), int(image.height * upscale)), Image.LANCZOS)
        return run_baseline(image, ocr_model, processor, field_model, model_fields,
                            ocr_mode=ocr_mode, stride=stride, tile=tile,
                            debug_dir=debug_dir, name=name)

    return run_rules(image, ocr_model, model_fields, tile=tile, target_long=target_long,
                     debug_dir=debug_dir, name=name)