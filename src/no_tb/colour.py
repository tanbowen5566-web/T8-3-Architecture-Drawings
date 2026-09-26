"""
src/no_tb/colour.py  (S1)

Ink colour of every OCR word, measured on the ORIGINAL (not enhanced) sheet.
Used by roles.py:
    red    small text at the sheet edge  -> archive annotation (A-18, 13, 3 ...)
    purple / blue                         -> rubber stamp (architect stamp, "DELIVERED AT ...")
    dark                                  -> ordinary drawing ink / pencil
Pink wall fills do not matter: only the darkest pixels inside a word box
(the strokes) are measured.
"""
import cv2
import numpy as np


def ink_colour(rgb: np.ndarray, px) -> str:
    x0, y0, x1, y1 = (int(round(v)) for v in px)
    crop = rgb[max(y0, 0):max(y1, 0), max(x0, 0):max(x1, 0)]
    if crop.size == 0:
        return "dark"
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    strokes = gray <= np.percentile(gray, 25)          # darkest quarter = the letters
    if strokes.sum() < 5 or gray.max() - gray.min() < 25:
        return "dark"
    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)[strokes]
    h, s = float(np.median(hsv[:, 0])), float(np.median(hsv[:, 1]))
    if s < 55:
        return "dark"
    if h < 10 or h >= 165:
        return "red"
    if h >= 125:
        return "purple"
    if h >= 90:
        return "blue"
    return "other"


def annotate(words, rgb: np.ndarray):
    for w in words:
        w["colour"] = ink_colour(rgb, w["px"])
    return words