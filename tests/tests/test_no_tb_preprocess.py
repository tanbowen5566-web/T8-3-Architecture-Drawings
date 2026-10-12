import numpy as np
from PIL import Image

from src.no_tb.preprocess import (
    normalise_scale,
    flatten_background,
    enhance,
)


def test_normalise_scale_no_resize():
    image = Image.new("RGB", (100, 80))

    scaled, factor = normalise_scale(
        image,
        target_long=100,
    )

    assert scaled.size == (100, 80)
    assert factor == 1.0


def test_normalise_scale_downscale():
    image = Image.new("RGB", (200, 100))

    scaled, factor = normalise_scale(
        image,
        target_long=100,
    )

    assert scaled.size == (100, 50)
    assert factor == 0.5


def test_normalise_scale_respects_max_up():
    image = Image.new("RGB", (100, 50))

    scaled, factor = normalise_scale(
        image,
        target_long=1000,
        max_up=2.0,
    )

    assert scaled.size == (200, 100)
    assert factor == 2.0


def test_flatten_background_returns_uint8():
    gray = np.full((100, 100), 200, dtype=np.uint8)
    gray[40:60, 40:60] = 80

    result = flatten_background(gray)

    assert result.shape == gray.shape
    assert result.dtype == np.uint8
    assert result.min() >= 0
    assert result.max() <= 255


def test_enhance_returns_image_and_mask():
    image = Image.new("RGB", (100, 80), "white")

    enhanced, ink = enhance(image)

    assert enhanced.size == image.size
    assert enhanced.mode == "RGB"

    assert ink.shape == (80, 100)
    assert ink.dtype == bool
