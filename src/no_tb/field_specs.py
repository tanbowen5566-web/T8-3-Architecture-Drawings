"""
src/no_tb/field_specs.py

All tunable vocabulary / patterns / thresholds for the rule-based No-TB
extractor in one place. Tuning = editing this file, not the logic.
Patterns were derived from BSM archival sheets (1968.0013.*): contract
sections/elevations and site plans without a title block.
"""
import re

# ---------------------------------------------------------------- view titles
# "SECTION A.B", "Ground Plan", "South Elevation", "LONGITUDINAL SECTION" ...
VIEW_WORDS = {"SECTION", "SECTIONS", "ELEVATION", "ELEVATIONS", "PLAN", "PLANS",
              "DETAIL", "DETAILS", "PERSPECTIVE", "VIEW"}
VIEW_MAX_WORDS = 7          # longer blocks with "PLAN" in them are notes, not view titles

# ---------------------------------------------------------------- scale
SCALE_RE = re.compile(
    r"\bSCALES?\b"
    r"|\b(?:FEET|FOOT|FT|LINKS?|CHAINS?|INCH(?:ES)?)\s+TO\s+(?:ONE|AN|1)\s+(?:INCH|FOOT)\b"
    r"|\d+(?:/\d+)?\s*(?:\"|”|''|IN\.?)\s*=\s*\d+\s*(?:'|’|FT)"
    r"|\b1\s*:\s*\d{1,5}\b", re.I)

NUMBER_WORDS = {
    "HALF": 0.5, "ONE": 1, "TWO": 2, "THREE": 3, "FOUR": 4, "FIVE": 5, "SIX": 6,
    "SEVEN": 7, "EIGHT": 8, "NINE": 9, "TEN": 10, "ELEVEN": 11, "TWELVE": 12,
    "SIXTEEN": 16, "TWENTY": 20, "THIRTY": 30, "FORTY": 40, "FIFTY": 50,
    "SIXTY": 60, "EIGHTY": 80, "HUNDRED": 100,
}
UNIT_INCHES = {"FOOT": 12, "FEET": 12, "FT": 12, "LINK": 7.92, "LINKS": 7.92,
               "CHAIN": 792, "CHAINS": 792, "INCH": 1, "INCHES": 1}


def _num(tok):
    tok = tok.upper().strip(".,")
    if tok in NUMBER_WORDS:
        return NUMBER_WORDS[tok]
    try:
        return float(tok)
    except ValueError:
        return None


def scale_to_ratio(text):
    """'EIGHT FEET TO ONE INCH' -> 96, '60 LINKS TO ONE INCH' -> 475.2, '1:100' -> 100.
    None if not parseable. Only used for confidence + debug, the output keeps the raw text."""
    t = text.upper()
    m = re.search(r"\b1\s*:\s*(\d{1,5})\b", t)
    if m:
        return float(m.group(1))
    m = re.search(r"([A-Z0-9./]+)\s+(FEET|FOOT|FT|LINKS?|CHAINS?|INCH(?:ES)?)\s+TO\s+(?:ONE|AN|1)\s+INCH", t)
    if m and _num(m.group(1)) is not None:
        return round(_num(m.group(1)) * UNIT_INCHES[m.group(2)], 2)
    return None


# ---------------------------------------------------------------- drawing number
# "Nº3", "No. 3", "DRAWING Nº 3", "Sheet 16". OCR reads º as o/0/°/9.
DRAWING_NO_RE = re.compile(
    r"(?:\b(?:DRAWING|DRG|DWG|PLAN)\.?\s*)?\bN\s?[O0º°o9]?\s*[.:]?\s*([A-Z]?\d{1,4}[A-Z]?)\b"
    r"|\bSHEET\s*(?:N[O0º°o]\.?)?\s*:?\s*([A-Z]?\d{1,4}[A-Z]?)\b", re.I)
# "Room No 10", "Lot No 4" are room / allotment labels, not the drawing number
NOT_DRAWING_NO_BEFORE = {"ROOM", "LOT", "ALLOTMENT", "SHOP", "FLAT", "HOUSE", "UNIT", "BAY", "BOX"}

# ---------------------------------------------------------------- stamps / contract notes
STAMP_WORDS = {"ARCHITECT", "ARCHITECTS", "SURVEYOR", "SURVEYORS", "ENGINEER", "ENGINEERS",
               "DELIVERED", "RECEIVED", "OFFICE", "APPROVED"}
ARCHITECT_WORDS = {"ARCHITECT", "ARCHITECTS", "ARCHT", "ARCHTS"}
# words removed from a stamp/label to leave the architect's name
ARCHITECT_NOISE = {"ARCHITECT", "ARCHITECTS", "ARCHT", "ARCHTS", "SURVEYOR",
                   "SURVEYORS", "ENGINEERS", "ENGINEER", "MELBOURNE", "ST", "STREET", "ST.",
                   "ARAIA", "FRAIA", "RAIA", "FRIBA", "ARIBA", "AIA", "LICENSED"}
CONTRACT_WORDS = {"CONTRACT", "CONTRACTOR", "WITNESS", "SIGNED", "REFERRED", "DATED",
                  "AGREEMENT", "DAY"}

# ---------------------------------------------------------------- draughtsperson labels
DRAUGHTSPERSON_LABELS = ["DRAUGHTSMAN", "DRAFTSMAN", "DRAWN BY", "TRACED BY", "DRAWN",
                         "TRACED", "DRN", "DELT", "DEL"]

# ---------------------------------------------------------------- dates
MONTHS = r"(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)[A-Z]*\.?"
DATE_RE = re.compile(
    rf"\b\d{{1,2}}(?:ST|ND|RD|TH)?\s+(?:DAY\s+OF\s+)?{MONTHS},?\s+(?:1[89]|20)\d{{2}}\b"
    rf"|\b{MONTHS}\s+\d{{1,2}}(?:ST|ND|RD|TH)?,?\s+(?:1[89]|20)\d{{2}}\b"
    rf"|\b{MONTHS},?\s+(?:1[89]|20)\d{{2}}\b"
    rf"|\b\d{{1,2}}\s*[./-]\s*\d{{1,2}}\s*[./-]\s*(?:1[89]|20)?\d{{2}}\b", re.I)
YEAR_RE = re.compile(r"\b(1[89]\d{2})\b")

# ---------------------------------------------------------------- noise
# dimension strings: 25'0", 14'-0", 116', 12"
DIMENSION_TOKEN_RE = re.compile(r"^\d{1,3}\s*['’]\s*-?\s*\d{0,2}\s*[\"”]?$|^\d{1,3}\s*[\"”]$|^\d+/\d+$|^\d{1,4}[.,]?$")
# archive marks: "A-18", "13", "828 F3", accession numbers 1968.0013.00301
ARCHIVE_RE = re.compile(r"^(?:[A-Z]{0,2}\s*-?\s*\d{1,4}[A-Z]?|\d{2,4}\s*[A-Z]\s*\d|1\d{3}[.\-\s]\d{4}[.\-\s]\d{3,5})$", re.I)
EDGE_MARGIN = 0.07          # fraction of the sheet treated as "edge" for archive marks

STREET_END = {"STREET", "ST", "ST.", "ROAD", "RD", "LANE", "AVENUE", "PARADE", "PLACE", "TERRACE"}

# ---------------------------------------------------------------- confidences (base scores)
CONF = {
    "title": 0.85, "view_title": 0.85, "scale": 0.9, "drawing_no": 0.85,
    "stamp_architect": 0.6, "label_value": 0.8, "contract_date": 0.6, "year_only": 0.35,
}