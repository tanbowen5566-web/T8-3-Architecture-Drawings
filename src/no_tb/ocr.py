"""
src/no_tb/ocr.py  (S2)

docTR on a full sheet, optionally tiled so small lettering is not downscaled
away (docTR's detector resizes every input to 1024x1024).

Word format used everywhere in src/no_tb:
    {"text": str, "conf": float,
     "px":   [x0, y0, x1, y1]  pixel coords in the image that was OCR'd,
     "bbox": [x0, y0, x1, y1]  same, normalised 0-1 (TB-branch convention)}
"""
import numpy as np

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


def reading_order(words):
    """Top-to-bottom, left-to-right, bucketed by typical text height."""
    if not words:
        return words
    mh = max(float(np.median([w["px"][3] - w["px"][1] for w in words])), 1.0)
    return sorted(words, key=lambda w: (round((w["px"][1] + w["px"][3]) / 2 / mh), w["px"][0]))