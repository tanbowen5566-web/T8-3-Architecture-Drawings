"""
src/no_tb/roles.py  (S4)

Give every text block exactly one role. Field mapping (extractor.py) only
reads blocks of the matching role, so street names can't become the project
name, room labels ("Room No 10") can't become the drawing number, and a
stamp date ("DELIVERED ... JAN 1906") can't become the drawing date.

Roles:
    title          project name (at most one per sheet, chosen by scoring)
    view_title     SECTION A.B / Ground Plan / South Elevation ...
    scale          SCALE EIGHT FEET TO ONE INCH / 1:100 ...
    drawing_no     Nº3 / DRAWING Nº3 / Sheet 16
    stamp          purple/blue ink or stamp wording (architect stamp, delivery stamp)
    contract_note  handwritten contract clause / signatures (bottom, low OCR confidence)
    archive_mark   red numbers / accession numbers at the sheet edge (added by the archive)
    dimension      25'0", 116', 12 ...
    unreadable     OCR confidence too low to use
    label          everything else (room names, street names, notes)
"""
import re

from . import field_specs as S


def tok(s):
    return re.sub(r"[^A-Z0-9&º°]", "", s.upper())


def _alpha(text):
    return sum(c.isalpha() for c in text)


def _is_edge(px, W, H, m=S.EDGE_MARGIN):
    return px[0] < m * W or px[2] > (1 - m) * W or px[1] < m * H or px[3] > (1 - m) * H


def drawing_no_match(block):
    """Returns the drawing number string or None. Skips 'Room No 10' style labels."""
    text = block["text"]
    for m in S.DRAWING_NO_RE.finditer(text):
        before = tok(text[:m.start()].split()[-1]) if text[:m.start()].split() else ""
        if before in S.NOT_DRAWING_NO_BEFORE:
            continue
        return (m.group(1) or m.group(2)).upper()
    return None


def _classify(b, W, H):
    text, toks = b["text"], [tok(w["text"]) for w in b["words"]]
    n_words = len(b["words"])
    cy = (b["px"][1] + b["px"][3]) / 2 / H
    edge = _is_edge(b["px"], W, H)
    compact = re.sub(r"\s+", " ", text).strip()

    if not re.search(r"[A-Za-z0-9]", text):
        return "unreadable", 0.0, "no letters/digits"
    if S.SCALE_RE.search(text):
        return "scale", 1.0, "scale pattern"
    if n_words <= 5 and drawing_no_match(b):
        score = 1.0 if (edge or re.search(r"DRAWING|SHEET|DRG|DWG", text, re.I)) else 0.7
        return "drawing_no", score, "No./Sheet + number"
    if b["colour"] == "red" and edge and len(compact) <= 14:
        return "archive_mark", 1.0, "red text at sheet edge"
    if edge and len(compact) <= 16 and S.ARCHIVE_RE.match(compact):
        return "archive_mark", 0.8, "short code at sheet edge"
    if b["colour"] in ("purple", "blue") or any(t in S.STAMP_WORDS for t in toks):
        return "stamp", 1.0, f"stamp ink/wording ({b['colour']})"
    if any(t in S.CONTRACT_WORDS for t in toks):
        return "contract_note", 1.0, "contract wording"
    if b["conf"] < 0.5 and cy > 0.6 and n_words >= 3:
        return "contract_note", 0.6, "low-confidence text in lower sheet (handwriting?)"
    if b["conf"] < 0.3:
        return "unreadable", 0.0, f"OCR conf {b['conf']:.2f}"
    if any(t in S.VIEW_WORDS for t in toks) and n_words <= S.VIEW_MAX_WORDS and _alpha(text) >= 4:
        return "view_title", 1.0, "view keyword"
    if all(S.DIMENSION_TOKEN_RE.match(w["text"].strip()) for w in b["words"]):
        return "dimension", 1.0, "dimension tokens"
    return "label", 0.0, ""


def _title_score(b, W, H, med_h):
    if _alpha(b["text"]) < 5 or b["conf"] < 0.4:
        return 0.0
    size = (b["h"] / med_h) ** 0.7
    iso = 1.0 / (1.0 + 40.0 * b["density"])          # empty paper around it -> ~1
    cy = (b["px"][1] + b["px"][3]) / 2 / H
    pos = 1.5 if cy < 0.35 else 1.0
    multi = 1.3 if len(b["lines"]) >= 2 else 1.0
    last = tok(b["words"][-1]["text"])
    street = 0.5 if (len(b["lines"]) == 1 and last in S.STREET_END) else 1.0
    return size * iso * pos * multi * street * b["conf"]


def assign_roles(blocks, W, H, med_h, density):
    for b in blocks:
        b["density"] = round(density(b["px"], margin=3 * b["h"]), 4)
        b["role"], b["role_score"], b["why"] = _classify(b, W, H)

    candidates = sorted(((b, _title_score(b, W, H, med_h)) for b in blocks if b["role"] == "label"),
                        key=lambda x: x[1], reverse=True)
    candidates = [c for c in candidates if c[1] > 0]
    if candidates:
        best, s1 = candidates[0]
        s2 = candidates[1][1] if len(candidates) > 1 else 0.0
        best["role"] = "title"
        best["role_score"] = round(s1, 3)
        best["title_margin"] = round((s1 - s2) / s1, 3) if s1 else 0.0
        best["why"] = f"title score {s1:.2f} (runner-up {s2:.2f})"
    return blocks