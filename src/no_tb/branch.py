"""
src/no_tb/branch.py

No-TB branch of run_pipeline.py: drawings where the title block detector finds
no high-confidence title block (score < DETECTION_SCORE_THRESHOLD).

Current implementation = "TB recipe on the full sheet" baseline:
    full image -> docTR -> the SAME LayoutLMv3 field classifier as the TB branch

Two differences from the TB branch, both needed because a whole sheet is not a crop:
  1. Sliding window over tokens. A full sheet easily exceeds 512 tokens; the TB
     branch simply truncates, which here would silently drop most of the sheet.
  2. One spatially-contiguous span per field (the most confident one) instead of
     joining every word with that label anywhere on the sheet -- otherwise e.g.
     all street names / room labels predicted as project_name get glued together.

process_no_tb() returns the same {field: {"value", "confidence"}} dict as
run_pipeline.predict_fields(), so both branches share all downstream code
(drawing_types keyword matching, save_output, confidence CSV).

Later, the rule/role-based No-TB extractor replaces the internals of
process_no_tb() without changing its signature.
"""
import json
import os
import re

import numpy as np
import torch
from PIL import Image, ImageDraw

Image.MAX_IMAGE_PIXELS = None  # archival scans can be very large


# ---------------------------------------------------------------------------
# OCR on the full sheet
# ---------------------------------------------------------------------------

def _page_words(page, ox, oy, cw, ch, W, H):
    """docTR page (possibly a tile) -> words with
    bbox: normalised 0-1 to the FULL image (same convention as the TB branch)
    px:   pixel coordinates in the full image (for spatial reasoning)"""
    items = []
    for block in page.blocks:
        for line in block.lines:
            for w in line.words:
                # works for straight ((x0,y0),(x1,y1)) and rotated (4-point) geometry
                pts = np.asarray(w.geometry, dtype=float).reshape(-1, 2)
                px = [ox + pts[:, 0].min() * cw, oy + pts[:, 1].min() * ch,
                      ox + pts[:, 0].max() * cw, oy + pts[:, 1].max() * ch]
                items.append({
                    "text": w.value,
                    "conf": float(w.confidence),
                    "px": px,
                    "bbox": [px[0] / W, px[1] / H, px[2] / W, px[3] / H],
                })
    return items


def _tile_starts(length, tile, step):
    if length <= tile:
        return [0]
    s = list(range(0, length - tile + 1, step))
    if s[-1] + tile < length:
        s.append(length - tile)
    return s


def _area(b):
    return max(b[2] - b[0], 0) * max(b[3] - b[1], 0)


def _dedupe(words, thr=0.5):
    """Remove duplicates from tile overlaps; keep the bigger (un-cut) box."""
    kept = []
    for w in sorted(words, key=lambda w: (_area(w["px"]), w["conf"]), reverse=True):
        a = w["px"]
        dup = False
        for k in kept:
            b = k["px"]
            ix = min(a[2], b[2]) - max(a[0], b[0])
            iy = min(a[3], b[3]) - max(a[1], b[1])
            if ix > 0 and iy > 0 and ix * iy > thr * min(_area(a), _area(b)):
                dup = True
                break
        if not dup:
            kept.append(w)
    return kept


def ocr_full_image(image, ocr_model, mode="full", tile=1024, overlap=128, batch=4):
    """mode="full":  one docTR pass on the whole sheet (identical to the TB branch)
    mode="tiled": overlapping tiles, so small lettering is not downscaled away"""
    arr = np.ascontiguousarray(np.array(image))
    H, W = arr.shape[:2]
    if mode == "full" or max(H, W) <= tile:
        return _page_words(ocr_model([arr]).pages[0], 0, 0, W, H, W, H)

    step = tile - overlap
    boxes = [(x, y, min(x + tile, W), min(y + tile, H))
             for y in _tile_starts(H, tile, step) for x in _tile_starts(W, tile, step)]
    words = []
    for i in range(0, len(boxes), batch):
        chunk = boxes[i:i + batch]
        res = ocr_model([np.ascontiguousarray(arr[y0:y1, x0:x1]) for x0, y0, x1, y1 in chunk])
        for (x0, y0, x1, y1), page in zip(chunk, res.pages):
            words += _page_words(page, x0, y0, x1 - x0, y1 - y0, W, H)
    return _dedupe(words)


def _reading_order(words):
    """Top-to-bottom, left-to-right, bucketed by typical text height."""
    if not words:
        return words
    mh = max(float(np.median([w["px"][3] - w["px"][1] for w in words])), 1.0)
    return sorted(words, key=lambda w: (round((w["px"][1] + w["px"][3]) / 2 / mh), w["px"][0]))


# ---------------------------------------------------------------------------
# LayoutLMv3 with a sliding window
# ---------------------------------------------------------------------------

def _to_1000(b):
    return [max(0, min(1000, int(v * 1000))) for v in b]


def _predict_words(image, words, processor, field_model, stride=128, max_length=512):
    """Returns [(label, prob)] per word, and the number of windows used."""
    id2label = field_model.config.id2label
    device = next(field_model.parameters()).device

    enc = processor(
        image, [w["text"] for w in words], boxes=[_to_1000(w["bbox"]) for w in words],
        truncation=True, padding="max_length", max_length=max_length, stride=stride,
        return_overflowing_tokens=True, return_offsets_mapping=True,
        return_tensors="pt",
    )
    n_win = enc["input_ids"].shape[0]
    word_ids = [enc.word_ids(batch_index=i) for i in range(n_win)]
    pixel_values = enc["pixel_values"]
    if isinstance(pixel_values, list):  # processor duplicates the image per window as a list
        pixel_values = torch.stack([p if torch.is_tensor(p) else torch.tensor(p) for p in pixel_values])

    best_prob = [0.0] * len(words)
    best_label = ["O"] * len(words)
    with torch.no_grad():
        for i in range(n_win):
            outputs = field_model(
                input_ids=enc["input_ids"][i:i + 1].to(device),
                attention_mask=enc["attention_mask"][i:i + 1].to(device),
                bbox=enc["bbox"][i:i + 1].to(device),
                pixel_values=pixel_values[i:i + 1].to(device),
            )
            probs = torch.softmax(outputs.logits, dim=-1).squeeze(0)
            preds = probs.argmax(-1).tolist()
            seen = set()
            for t, wid in enumerate(word_ids[i]):
                if wid is None or wid in seen:
                    continue
                seen.add(wid)  # first sub-token decides the word (same as TB branch)
                p = probs[t][preds[t]].item()
                if p > best_prob[wid]:  # overlapping windows: keep the most confident prediction
                    best_prob[wid] = p
                    best_label[wid] = id2label[preds[t]]
    return list(zip(best_label, best_prob)), n_win


# ---------------------------------------------------------------------------
# word labels -> one value per field
# ---------------------------------------------------------------------------

def _field_of(label, model_fields):
    """'architect' / 'B-architect' / 'I-architect' -> 'architect'; anything else -> None."""
    if not label:
        return None
    m = re.match(r"^[BIES][-_](.+)$", label)
    name = m.group(1) if m else label
    return name if name in model_fields else None


def _adjacent(a, b):
    """b follows a in reading order AND is spatially close (same line, or the line just below)."""
    ax0, ay0, ax1, ay1 = a["px"]
    bx0, by0, bx1, by1 = b["px"]
    h = max(ay1 - ay0, by1 - by0, 1.0)
    same_row = abs((ay0 + ay1) / 2 - (by0 + by1) / 2) < 0.6 * h and -0.5 * h <= bx0 - ax1 <= 3 * h
    next_row = 0 <= by0 - ay1 <= 1.5 * h and bx0 < ax1 + 3 * h and bx1 > ax0 - 3 * h
    return same_row or next_row


def _pick_fields(words, preds, model_fields):
    spans, cur = [], None
    for w, (label, prob) in zip(words, preds):
        field = _field_of(label, model_fields)
        if field is None:
            cur = None
            continue
        starts_new = label.startswith(("B-", "B_", "S-", "S_"))
        if cur is None or starts_new or cur["field"] != field or not _adjacent(cur["words"][-1], w):
            cur = {"field": field, "words": [], "probs": []}
            spans.append(cur)
        cur["words"].append(w)
        cur["probs"].append(prob)

    result = {f: {"value": "", "confidence": None} for f in model_fields}
    best_score = {}
    for s in spans:
        conf = sum(s["probs"]) / len(s["probs"])
        score = (conf, len(s["words"]))  # tie-break: longer span
        if s["field"] not in best_score or score > best_score[s["field"]]:
            best_score[s["field"]] = score
            result[s["field"]] = {"value": " ".join(w["text"] for w in s["words"]), "confidence": conf}
    return result, len(spans)


# ---------------------------------------------------------------------------
# debug dump: tells you whether a miss is OCR (word not there) or model (word there, labelled O)
# ---------------------------------------------------------------------------

def _dump_debug(debug_dir, name, image, words, preds, fields, info):
    os.makedirs(debug_dir, exist_ok=True)
    stem = os.path.splitext(name)[0]
    rows = [{"text": w["text"], "ocr_conf": round(w["conf"], 3), "label": lab, "prob": round(p, 3),
             "px": [round(v) for v in w["px"]]} for w, (lab, p) in zip(words, preds)]
    with open(os.path.join(debug_dir, f"{stem}.json"), "w", encoding="utf-8") as f:
        json.dump({"file": name, "size": list(image.size), **info, "fields": fields,
                   "words": rows}, f, indent=1, ensure_ascii=False)

    im = image.copy()
    d = ImageDraw.Draw(im)
    lw = max(1, im.width // 1200)
    for w, (lab, p) in zip(words, preds):
        labelled = lab not in ("O", "", None)
        d.rectangle(w["px"], outline=(220, 0, 0) if labelled else (0, 120, 255), width=lw * (3 if labelled else 1))
        if labelled:
            d.text((w["px"][0], max(w["px"][1] - 14 * lw, 0)), f"{lab} {p:.2f}", fill=(220, 0, 0))
    if max(im.size) > 4096:
        k = 4096 / max(im.size)
        im = im.resize((int(im.width * k), int(im.height * k)))
    im.save(os.path.join(debug_dir, f"{stem}_overlay.jpg"), quality=88)


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------

def process_no_tb(image_path, ocr_model, processor, field_model, model_fields,
                  ocr_mode="full", upscale=1.0, tile=1024, stride=128, debug_dir=None):
    """Full-sheet extraction for a drawing without a detected title block.

    Returns (fields, info)
      fields: {field: {"value": str, "confidence": float | None}}  -- same as predict_fields()
      info:   {"n_words", "n_windows", "n_spans"}                    -- for logging / QA
    fields is None if OCR found no text.
    debug_dir: if set, writes <name>.json (every OCR word + predicted label) and
               <name>_overlay.jpg (blue = word labelled O, red = labelled field).
    """
    image = Image.open(image_path).convert("RGB")
    if upscale != 1.0:
        image = image.resize((int(image.width * upscale), int(image.height * upscale)), Image.LANCZOS)

    words = _reading_order(ocr_full_image(image, ocr_model, mode=ocr_mode, tile=tile))
    if not words:
        if debug_dir:
            _dump_debug(debug_dir, os.path.basename(str(image_path)), image, [], [], None,
                        {"n_words": 0, "n_windows": 0, "n_spans": 0})
        return None, {"n_words": 0, "n_windows": 0, "n_spans": 0}

    preds, n_win = _predict_words(image, words, processor, field_model, stride=stride)
    fields, n_spans = _pick_fields(words, preds, model_fields)
    info = {"n_words": len(words), "n_windows": n_win, "n_spans": n_spans,
            "n_labelled_words": sum(1 for lab, _ in preds if _field_of(lab, model_fields))}
    if debug_dir:
        _dump_debug(debug_dir, os.path.basename(str(image_path)), image, words, preds, fields,
                    {**info, "ocr_mode": ocr_mode, "upscale": upscale})
    return fields, info