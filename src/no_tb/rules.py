"""
src/no_tb/rules.py

Rule-based No-TB pipeline (default method):
    S0 preprocess  -> S2 tiled OCR -> S1 ink colour -> S3 layout
    -> S4 roles -> S5/S7 fields + status  (+ debug dump)
"""
import json
import os
from statistics import median

import numpy as np
from PIL import ImageDraw

from . import layout, preprocess, colour
from .extractor import extract
from .ocr import ocr_full_image
from .roles import assign_roles

ROLE_COLOURS = {
    "title": (220, 0, 0), "view_title": (230, 120, 0), "scale": (0, 150, 0),
    "drawing_no": (150, 0, 200), "stamp": (200, 0, 150), "contract_note": (0, 90, 200),
    "archive_mark": (120, 120, 120), "dimension": (170, 170, 170),
    "unreadable": (200, 200, 200), "label": (0, 170, 220),
}


def run_rules(image, ocr_model, model_fields, tile=1024, target_long=4096, ocr_on="enhanced",
              debug_dir=None, name="image"):
    # S0
    scaled, factor = preprocess.normalise_scale(image, target_long=target_long)
    enhanced, ink = preprocess.enhance(scaled)
    ocr_input = enhanced if ocr_on == "enhanced" else scaled
    W, H = scaled.size

    # S2 (tiled when the sheet is larger than one tile)
    words = ocr_full_image(ocr_input, ocr_model, mode="tiled", tile=tile)
    words = [w for w in words if w["text"].strip()]
    info = {"n_words": len(words), "scale_factor": round(factor, 3), "size": [W, H]}
    if not words:
        if debug_dir:
            _dump(debug_dir, name, scaled, [], None, None, info)
        return None, info

    # S1 + S3
    colour.annotate(words, np.asarray(scaled))
    words = layout.merge_spaced_letters(words)
    blocks = layout.group_blocks(layout.group_lines(words))
    med_h = median(b["h"] for b in blocks)
    density = layout.InkDensity(ink, words)

    # S4 + S5
    assign_roles(blocks, W, H, med_h, density)
    fields, status = extract(blocks, model_fields)

    roles = {}
    for b in blocks:
        roles[b["role"]] = roles.get(b["role"], 0) + 1
    info.update({"n_blocks": len(blocks), "roles": roles,
                 "found": sum(s == "found" for s in status.values())})
    if debug_dir:
        _dump(debug_dir, name, scaled, blocks, fields, status, info)
    return fields, info


def _dump(debug_dir, name, image, blocks, fields, status, info):
    os.makedirs(debug_dir, exist_ok=True)
    stem = os.path.splitext(name)[0]
    rows = [{k: b[k] for k in ("id", "role", "role_score", "why", "text", "lines", "colour",
                                "conf", "h", "density") if k in b}
            | {"px": [round(v) for v in b["px"]]}
            | ({"scale_ratio": b["scale_ratio"]} if "scale_ratio" in b else {})
            for b in blocks]
    with open(os.path.join(debug_dir, f"{stem}.json"), "w", encoding="utf-8") as f:
        json.dump({"file": name, "method": "rules", **info, "fields": fields, "status": status,
                   "blocks": sorted(rows, key=lambda r: r["role"])}, f, indent=1, ensure_ascii=False,
                  default=float)

    im = image.convert("RGB").copy()
    d = ImageDraw.Draw(im)
    lw = max(2, im.width // 1000)
    for b in blocks:
        c = ROLE_COLOURS.get(b["role"], (0, 0, 0))
        thick = b["role"] not in ("label", "dimension", "unreadable", "archive_mark")
        d.rectangle(b["px"], outline=c, width=lw * (3 if thick else 1))
        if thick:
            d.text((b["px"][0], max(b["px"][1] - 14 * lw, 0)), b["role"], fill=c)
    if max(im.size) > 4096:
        k = 4096 / max(im.size)
        im = im.resize((int(im.width * k), int(im.height * k)))
    im.save(os.path.join(debug_dir, f"{stem}_overlay.jpg"), quality=88)