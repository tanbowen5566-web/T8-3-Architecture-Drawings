"""
src/no_tb/layout.py  (S3)

words -> (merge letter-spaced capitals) -> lines -> blocks, plus per-block
layout features used by roles.py.

Block = a group of vertically adjacent lines with similar lettering size,
e.g. the 3-line title "CHRISTIAN INSTITUTE / INDEPENDENT CHURCH / COLLINS STREET".
"""
from statistics import median

import numpy as np


def _h(w):
    return max(w["px"][3] - w["px"][1], 1.0)


def _cy(w):
    return (w["px"][1] + w["px"][3]) / 2


def _union(ws):
    return [min(w["px"][0] for w in ws), min(w["px"][1] for w in ws),
            max(w["px"][2] for w in ws), max(w["px"][3] for w in ws)]


def _merge_words(ws, sep=""):
    px = _union(ws)
    return {"text": sep.join(w["text"] for w in ws),
            "conf": float(np.mean([w["conf"] for w in ws])),
            "px": px,
            "colour": ws[0].get("colour", "dark"),
            "spaced": True}   # letter-spaced lettering: word gaps are wider than usual


# ---------------------------------------------------------------- letter-spaced capitals
def _rows(words, max_gap_h=3.0):
    """Greedy left-to-right chaining of words on the same baseline."""
    rows = []
    for w in sorted(words, key=lambda w: w["px"][0]):
        best, best_gap = None, None
        for r in rows:
            last = r[-1]
            h = max(_h(last), _h(w))
            gap = w["px"][0] - last["px"][2]
            lim = max(max_gap_h, 3.5) if (last.get("spaced") and w.get("spaced")) else max_gap_h
            if abs(_cy(last) - _cy(w)) < 0.5 * h and -0.3 * h <= gap <= lim * h \
                    and max(_h(last), _h(w)) <= 2 * min(_h(last), _h(w)):
                if best_gap is None or gap < best_gap:
                    best, best_gap = r, gap
        if best is not None:
            best.append(w)
        else:
            rows.append([w])
    return rows


def merge_spaced_letters(words):
    """'C H R I S T I A N   I N S T I T U T E' OCR'd letter by letter -> 'CHRISTIAN', 'INSTITUTE'.
    Word breaks = gaps clearly larger than the typical letter gap in that run."""
    is_single = lambda w: len(w["text"]) == 1 and w["text"].isalpha()
    singles = [w for w in words if is_single(w)]
    out = [w for w in words if not is_single(w)]
    for row in _rows(singles):
        if len(row) < 3:
            out += row
            continue
        gaps = [b["px"][0] - a["px"][2] for a, b in zip(row, row[1:])]
        thr = 1.6 * max(median(gaps), 1.0)
        group = [row[0]]
        for g, w in zip(gaps, row[1:]):
            if g > thr:
                out.append(_merge_words(group) if len(group) > 1 else group[0])
                group = [w]
            else:
                group.append(w)
        out.append(_merge_words(group) if len(group) > 1 else group[0])
    return out


# ---------------------------------------------------------------- lines and blocks
def group_lines(words):
    lines = []
    for row in _rows(words, max_gap_h=1.8):
        lines.append({"words": row, "px": _union(row), "h": median(_h(w) for w in row),
                      "text": " ".join(w["text"] for w in row)})
    return sorted(lines, key=lambda l: (l["px"][1], l["px"][0]))


def _x_related(a, b):
    ax0, _, ax1, _ = a["px"]
    bx0, _, bx1, _ = b["px"]
    overlap = min(ax1, bx1) - max(ax0, bx0)
    centre = abs((ax0 + ax1) / 2 - (bx0 + bx1) / 2)
    return overlap > 0 or centre < 0.25 * max(ax1 - ax0, bx1 - bx0)


def group_blocks(lines, line_sep=", "):
    blocks = []
    for ln in lines:
        target = None
        for b in blocks:
            last = b["lines"][-1]
            h = max(last["h"], ln["h"])
            gap = ln["px"][1] - last["px"][3]
            if -0.3 * h <= gap <= 1.2 * h and max(last["h"], ln["h"]) <= 1.8 * min(last["h"], ln["h"]) \
                    and _x_related(last, ln):
                target = b
                break
        if target is None:
            blocks.append({"lines": [ln]})
        else:
            target["lines"].append(ln)

    out = []
    for i, b in enumerate(blocks):
        words = [w for l in b["lines"] for w in l["words"]]
        colours = [w.get("colour", "dark") for w in words]
        out.append({
            "id": i,
            "lines": [l["text"] for l in b["lines"]],
            "text": line_sep.join(l["text"] for l in b["lines"]),
            "words": words,
            "px": _union(words),
            "h": median(_h(w) for w in words),
            "conf": float(np.mean([w["conf"] for w in words])),
            "colour": max(set(colours), key=colours.count),
        })
    return out


# ---------------------------------------------------------------- surrounding ink density
class InkDensity:
    """Fraction of ink pixels around a block, with all text boxes erased first.
    Low = block sits on empty paper (titles, view titles, notes);
    high = block sits inside the drawing (room labels, dimensions)."""

    def __init__(self, ink: np.ndarray, words):
        ink = ink.copy()
        for w in words:
            x0, y0, x1, y1 = (int(round(v)) for v in w["px"])
            pad = int(0.2 * _h(w))
            ink[max(y0 - pad, 0):y1 + pad, max(x0 - pad, 0):x1 + pad] = False
        self.H, self.W = ink.shape
        self.ii = np.pad(ink.astype(np.int64).cumsum(0).cumsum(1), ((1, 0), (1, 0)))

    def __call__(self, px, margin):
        x0 = int(max(px[0] - margin, 0)); y0 = int(max(px[1] - margin, 0))
        x1 = int(min(px[2] + margin, self.W)); y1 = int(min(px[3] + margin, self.H))
        area = max((x1 - x0) * (y1 - y0), 1)
        s = self.ii[y1, x1] - self.ii[y0, x1] - self.ii[y1, x0] + self.ii[y0, x0]
        return float(s) / area