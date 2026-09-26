"""
src/no_tb/baseline.py

"TB recipe on the full sheet" baseline, kept for comparison:
    full image -> docTR -> the SAME LayoutLMv3 field classifier as the TB branch
Selected with  run_pipeline.py --no_tb_method layoutlm

Two differences from the TB branch, both needed because a whole sheet is not a crop:
  1. Sliding window over tokens (a full sheet easily exceeds 512 tokens).
  2. One spatially-contiguous span per field (the most confident one) instead of
     joining every word with that label anywhere on the sheet.
"""
import json
import os
import re

import torch
from PIL import ImageDraw

from .ocr import ocr_full_image, reading_order

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



def run_baseline(image, ocr_model, processor, field_model, model_fields,
                 ocr_mode="full", stride=128, tile=1024, debug_dir=None, name="image"):
    words = reading_order(ocr_full_image(image, ocr_model, mode=ocr_mode, tile=tile))
    if not words:
        if debug_dir:
            _dump_debug(debug_dir, name, image, [], [], None, {"n_words": 0})
        return None, {"n_words": 0, "n_windows": 0, "n_spans": 0, "n_labelled_words": 0}
    preds, n_win = _predict_words(image, words, processor, field_model, stride=stride)
    fields, n_spans = _pick_fields(words, preds, model_fields)
    info = {"n_words": len(words), "n_windows": n_win, "n_spans": n_spans,
            "n_labelled_words": sum(1 for lab, _ in preds if _field_of(lab, model_fields))}
    if debug_dir:
        _dump_debug(debug_dir, name, image, words, preds, fields, {**info, "ocr_mode": ocr_mode})
    return fields, info