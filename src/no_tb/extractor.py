"""
src/no_tb/extractor.py  (S5 + S7)

Role-tagged blocks -> the MODEL_FIELDS dict run_pipeline expects:
    {field: {"value": str, "confidence": float | None}}
plus a per-field status for QA:  found / not_present
(an empty value is always confidence None, like the TB branch).
"""
import re

from . import field_specs as S
from .roles import drawing_no_match, tok


def _clean(text):
    text = re.sub(r"[_—–]{1,}", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" -:;,.")


def _reading(blocks):
    return sorted(blocks, key=lambda b: (round(b["px"][1] / max(b["h"], 1) / 3), b["px"][0]))


def _project_name(blocks):
    t = [b for b in blocks if b["role"] == "title"]
    if not t:
        return None
    b = t[0]
    conf = S.CONF["title"] * b["conf"] * (0.7 + 0.3 * b.get("title_margin", 0.0))
    return _clean(b["text"]), conf


def _drawing_name(blocks):
    views = _reading([b for b in blocks if b["role"] == "view_title"])
    names, seen = [], set()
    for b in views:
        name = _clean(b["text"])
        if name and name.upper() not in seen:
            seen.add(name.upper())
            names.append(name)
    if not names:
        return None
    conf = S.CONF["view_title"] * sum(b["conf"] for b in views) / len(views)
    return "; ".join(names), conf


def _drawing_number(blocks):
    cands = [b for b in blocks if b["role"] == "drawing_no"]
    if not cands:
        return None
    b = max(cands, key=lambda b: (b["role_score"], b["conf"]))
    return drawing_no_match(b), S.CONF["drawing_no"] * b["conf"] * b["role_score"]


def _scale(blocks):
    cands = [b for b in blocks if b["role"] == "scale"]
    if not cands:
        return None
    b = max(cands, key=lambda b: (bool(re.search(r"\bSCALE", b["text"], re.I)), b["conf"]))
    text = re.sub(r"^.*?\bSCALES?\b", "", b["text"], flags=re.I) or b["text"]
    text = _clean(text)
    ratio = S.scale_to_ratio(text)
    b["scale_ratio"] = ratio
    return text, S.CONF["scale"] * b["conf"] * (1.0 if ratio else 0.75)


def _date(blocks):
    # stamps are excluded on purpose: a "DELIVERED ... JAN 1906" stamp date is not the drawing date
    notes = [b for b in blocks if b["role"] == "contract_note"]
    others = [b for b in blocks if b["role"] not in ("stamp", "contract_note", "archive_mark")
              and re.search(r"\bDATE", b["text"], re.I)]
    for group, base in ((notes, S.CONF["contract_date"]), (others, S.CONF["label_value"])):
        for b in group:
            m = S.DATE_RE.search(b["text"])
            if m:
                return _clean(m.group(0)), base * b["conf"]
    for b in notes:
        m = S.YEAR_RE.search(b["text"])
        if m:
            return m.group(1), S.CONF["year_only"] * b["conf"]
    return None


def _strip_architect_noise(words):
    toks = [w["text"] for w in words]
    keep = []
    for i, t in enumerate(toks):
        nxt = tok(toks[i + 1]) if i + 1 < len(toks) else ""
        if tok(t) in S.ARCHITECT_NOISE or re.search(r"\d", t) or nxt in ("ST", "STREET"):
            continue
        keep.append(t)
    while keep and keep[0].lower() in ("&", "and"):
        keep.pop(0)
    while keep and keep[-1].lower() in ("&", "and"):
        keep.pop()
    keep = [t for i, t in enumerate(keep) if not (t == "&" and i > 0 and keep[i - 1] == "&")]
    return _clean(" ".join(keep))


def _near(a, b, k=4.0):
    m = k * max(a["h"], b["h"])
    return not (a["px"][2] + m < b["px"][0] or b["px"][2] + m < a["px"][0]
                or a["px"][3] + m < b["px"][1] or b["px"][3] + m < a["px"][1])


def _architect(blocks):
    for b in blocks:
        toks = [tok(w["text"]) for w in b["words"]]
        if any(t in S.ARCHITECT_WORDS for t in toks):
            words = b["words"]
            if b["role"] == "stamp":
                # a stamp is often split into several blocks (name line is bigger / rotated):
                # read all stamp blocks next to this one together
                cluster = _reading([s for s in blocks if s["role"] == "stamp" and (s is b or _near(s, b))])
                words = [w for s in cluster for w in s["words"]]
            name = _strip_architect_noise(words)
            if len(re.sub(r"[^A-Za-z]", "", name)) >= 3:
                base = S.CONF["stamp_architect"] if b["role"] == "stamp" else S.CONF["label_value"]
                return name, base * b["conf"]
    return None


def _draughtsperson(blocks):
    labels = "|".join(re.escape(l) for l in S.DRAUGHTSPERSON_LABELS)
    for b in blocks:
        if b["role"] in ("stamp", "archive_mark"):
            continue
        m = re.search(rf"\b(?:{labels})\b\.?\s*:?\s*(.+)", b["text"], re.I)
        if m and re.search(r"[A-Za-z]", m.group(1)):
            return _clean(m.group(1))[:40], S.CONF["label_value"] * b["conf"]
    return None


EXTRACTORS = {
    "project_name": _project_name,
    "drawing_name": _drawing_name,
    "drawing_number": _drawing_number,
    "scale": _scale,
    "date": _date,
    "architect": _architect,
    "draughtsperson": _draughtsperson,
}


def extract(blocks, model_fields):
    fields, status = {}, {}
    for f in model_fields:
        res = EXTRACTORS[f](blocks) if f in EXTRACTORS else None
        if res and res[0]:
            fields[f] = {"value": res[0], "confidence": round(min(max(res[1], 0.0), 1.0), 4)}
            status[f] = "found"
        else:
            fields[f] = {"value": "", "confidence": None}
            status[f] = "not_present"
    return fields, status