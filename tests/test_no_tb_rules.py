"""
Unit tests for the rule-based No-TB extractor (S1 colour not needed, S3-S5 only).
No docTR / torch needed: sheets are simulated as OCR word lists modelled on
BSM sheets 1968.0013.00368 (contract section) and 00301 (site plan).

    pytest tests/test_no_tb_rules.py -q
"""
import sys
import types
from statistics import median

import numpy as np

sys.modules.setdefault("torch", types.ModuleType("torch"))  # baseline.py imports torch

from src.no_tb import layout  # noqa: E402
from src.no_tb.extractor import extract  # noqa: E402
from src.no_tb.field_specs import scale_to_ratio  # noqa: E402
from src.no_tb.roles import assign_roles  # noqa: E402

FIELDS = ["architect", "project_name", "draughtsperson", "date", "scale", "drawing_number", "drawing_name"]
W, H = 4096, 2982


def words_from(text, x, y, h, conf=0.95, colour="dark", spaced=False):
    """Lay a text line out as OCR words (optionally letter-spaced, one word per letter)."""
    out, cx = [], x
    for token in text.split(" "):
        if spaced:
            for ch in token:
                out.append({"text": ch, "conf": conf, "px": [cx, y, cx + 0.6 * h, y + h], "colour": colour})
                cx += 0.6 * h + 0.5 * h
            cx += 1.5 * h
        else:
            wd = max(len(token), 1) * 0.6 * h
            out.append({"text": token, "conf": conf, "px": [cx, y, cx + wd, y + h], "colour": colour})
            cx += wd + 0.4 * h
    return out


def run(words, drawing_boxes):
    ink = np.zeros((H, W), bool)
    for x0, y0, x1, y1 in drawing_boxes:                 # hatch = "the drawing"
        ink[y0:y1:12, x0:x1] = True
        ink[y0:y1, x0:x1:12] = True
    words = layout.merge_spaced_letters(words)
    blocks = layout.group_blocks(layout.group_lines(words))
    med_h = median(b["h"] for b in blocks)
    assign_roles(blocks, W, H, med_h, layout.InkDensity(ink, words))
    fields, status = extract(blocks, FIELDS)
    return {f: v["value"] for f, v in fields.items()}, blocks


def test_contract_section_sheet():
    w = []
    w += words_from("CHRISTIAN INSTITUTE", 440, 210, 40, spaced=True)
    w += words_from("INDEPENDENT CHURCH", 500, 290, 40)
    w += words_from("COLLINS STREET", 620, 370, 40)
    w += words_from("No3", 3700, 250, 60)
    w += words_from("SECTION A.B", 1160, 1450, 30)
    w += words_from("SECTION E.F", 2760, 1450, 30)
    w += words_from("SECTION C.D", 540, 2710, 30)
    w += words_from("SCALE _ EIGHT FEET TO ONE INCH", 60, 2810, 28)
    w += words_from("This is one of the drawings referred to in my Contract dated November 6th 1871",
                    2540, 2600, 34, conf=0.35)
    w += words_from("25'0\"", 1260, 900, 26)
    w += words_from("14'0\"", 3000, 800, 26)
    w += words_from("3", 3900, 2900, 40, colour="red")
    boxes = [(900, 500, 1700, 1400), (2400, 300, 3300, 1400), (900, 1650, 3000, 2600)]
    v, blocks = run(w, boxes)

    assert v["project_name"] == "CHRISTIAN INSTITUTE, INDEPENDENT CHURCH, COLLINS STREET"
    assert v["drawing_number"] == "3"
    assert v["drawing_name"] == "SECTION A.B; SECTION E.F; SECTION C.D"
    assert v["scale"] == "EIGHT FEET TO ONE INCH"
    assert v["date"] == "November 6th 1871"
    assert v["architect"] == "" and v["draughtsperson"] == ""
    assert {b["text"]: b["role"] for b in blocks}["3"] == "archive_mark"


def test_site_plan_sheet():
    w = []
    w += words_from("CONVENT OF THE GOOD SHEPHERD", 1200, 360, 60)
    w += words_from("ABBOTSFORD", 1900, 470, 50)
    w += words_from("SCALE 60 LINKS TO ONE INCH", 2900, 1030, 28)
    w += words_from("CLARKE STREET", 250, 1400, 90)                     # big street name
    w += words_from("LAUNDRY", 3450, 1650, 26)
    w += words_from("Room No 10", 1000, 1700, 26)                       # must not be the drawing number
    w += words_from("DRYING YARD", 3050, 1950, 34)
    w += words_from("Architects & Surveyors", 3480, 2640, 26, colour="purple")
    w += words_from("REED, SMART & TAPPIN", 3470, 2720, 34, colour="purple")
    w += words_from("11 Elizabeth St Melbourne", 3500, 2790, 26, colour="purple")
    w += words_from("Sheet 16", 3200, 2860, 40)
    w += words_from("828 F3", 3960, 2930, 30)
    boxes = [(700, 1000, 1700, 2000), (3300, 1500, 3800, 2600), (2950, 1900, 3300, 2100)]
    v, _ = run(w, boxes)

    assert v["project_name"] == "CONVENT OF THE GOOD SHEPHERD, ABBOTSFORD"
    assert v["drawing_number"] == "16"
    assert v["scale"] == "60 LINKS TO ONE INCH"
    assert v["architect"] == "REED, SMART & TAPPIN"
    assert v["drawing_name"] == ""  # site plan has no view title -> left for review


def test_scale_ratio():
    assert scale_to_ratio("EIGHT FEET TO ONE INCH") == 96
    assert scale_to_ratio("60 LINKS TO ONE INCH") == 475.2
    assert scale_to_ratio("1:100") == 100
    assert scale_to_ratio("NOT A SCALE") is None


def test_spaced_letters_merge():
    ws = layout.merge_spaced_letters(words_from("CHRISTIAN INSTITUTE", 0, 0, 40, spaced=True))
    assert sorted(w["text"] for w in ws) == ["CHRISTIAN", "INSTITUTE"]