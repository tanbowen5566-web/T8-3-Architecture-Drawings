"""
src/no_tb/preprocess.py  (S0)

Archival No-TB scans come in very different sizes (2048 px derivatives up to
6000+ px masters) with faint grey lettering on yellowed paper.

normalise_scale(): bring the long side to ~target so lettering ends up a
                   similar pixel height on every sheet (then OCR is tiled at 1024,
                   so docTR never downscales it again).
enhance():         flatten the paper background (divide by a large-scale median),
                   then mild CLAHE -> faint lettering gets real contrast.
                   Returns the RGB image to OCR + a boolean ink mask used for
                   layout features (how "busy" the area around a text block is).
"""
import cv2
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def normalise_scale(image: Image.Image, target_long=4096, max_up=2.0):
    """Returns (scaled image, factor). Never upscales more than max_up."""
    factor = min(max_up, target_long / max(image.size))
    if abs(factor - 1.0) < 0.05:
        return image, 1.0
    size = (int(round(image.width * factor)), int(round(image.height * factor)))
    return image.resize(size, Image.LANCZOS if factor < 1 else Image.BICUBIC), factor


def flatten_background(gray: np.ndarray) -> np.ndarray:
    """gray / estimated paper background -> paper ~255, ink stays dark. uint8."""
    H, W = gray.shape
    small = cv2.resize(gray, (max(W // 8, 1), max(H // 8, 1)), interpolation=cv2.INTER_AREA)
    k = max(21, (min(small.shape) // 20) | 1)          # odd kernel, ~5% of the page
    bg = cv2.medianBlur(small, min(k, 99))
    bg = cv2.resize(bg, (W, H), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    flat = gray.astype(np.float32) / np.maximum(bg, 1.0) * 255.0
    return np.clip(flat, 0, 255).astype(np.uint8)


def enhance(image: Image.Image, clahe_clip=1.5, ink_threshold=170):
    """Returns (enhanced RGB PIL image for OCR, ink mask bool HxW)."""
    arr = np.asarray(image.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    flat = flatten_background(gray)
    out = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8)).apply(flat)
    ink = flat < ink_threshold
    return Image.fromarray(cv2.cvtColor(out, cv2.COLOR_GRAY2RGB)), ink