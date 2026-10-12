from unittest.mock import patch

import numpy as np
from PIL import Image

from src.no_tb.rules import run_rules, _dump


def test_run_rules_no_ocr_words():
    image = Image.new("RGB", (100, 80))

    with patch(
        "src.no_tb.rules.preprocess.normalise_scale",
        return_value=(image, 0.5),
    ), patch(
        "src.no_tb.rules.preprocess.enhance",
        return_value=(image, np.zeros((80, 100), dtype=bool)),
    ), patch(
        "src.no_tb.rules.ocr_full_image",
        return_value=[],
    ):
        fields, info = run_rules(
            image=image,
            ocr_model=None,
            model_fields=["drawing_name"],
        )

    assert fields is None
    assert info["n_words"] == 0
    assert info["scale_factor"] == 0.5
    assert info["size"] == [100, 80]


def test_run_rules_successful_extraction():
    image = Image.new("RGB", (100, 80))

    words = [
        {
            "text": "PLAN",
            "bbox": [0, 0, 1, 1],
        }
    ]

    blocks = [
        {
            "h": 10,
            "role": "view_title",
            "text": "PLAN",
        }
    ]

    expected_fields = {
        "drawing_name": {
            "value": "PLAN",
            "confidence": 0.9,
        }
    }

    status = {
        "drawing_name": "found",
    }

    with patch(
        "src.no_tb.rules.preprocess.normalise_scale",
        return_value=(image, 1.0),
    ), patch(
        "src.no_tb.rules.preprocess.enhance",
        return_value=(image, np.zeros((80, 100), dtype=bool)),
    ), patch(
        "src.no_tb.rules.ocr_full_image",
        return_value=words,
    ), patch(
        "src.no_tb.rules.colour.annotate",
    ), patch(
        "src.no_tb.rules.layout.merge_spaced_letters",
        return_value=words,
    ), patch(
        "src.no_tb.rules.layout.group_lines",
        return_value=["line"],
    ), patch(
        "src.no_tb.rules.layout.group_blocks",
        return_value=blocks,
    ), patch(
        "src.no_tb.rules.layout.InkDensity",
        return_value=object(),
    ), patch(
        "src.no_tb.rules.assign_roles",
    ), patch(
        "src.no_tb.rules.extract",
        return_value=(expected_fields, status),
    ):
        fields, info = run_rules(
            image=image,
            ocr_model=None,
            model_fields=["drawing_name"],
        )

    assert fields == expected_fields
    assert info["n_words"] == 1
    assert info["n_blocks"] == 1
    assert info["roles"]["view_title"] == 1
    assert info["found"] == 1


def test_dump_creates_debug_files(tmp_path):
    image = Image.new("RGB", (200, 100))

    blocks = [
        {
            "id": 1,
            "role": "view_title",
            "role_score": 0.9,
            "why": "test",
            "text": "SECOND FLOOR PLAN",
            "lines": [],
            "colour": "dark",
            "conf": 0.95,
            "h": 20,
            "density": 0.5,
            "px": [10, 10, 150, 40],
            "scale_ratio": 100,
        }
    ]

    fields = {
        "drawing_name": {
            "value": "SECOND FLOOR PLAN",
            "confidence": 0.95,
        }
    }

    status = {
        "drawing_name": "found",
    }

    info = {
        "n_words": 3,
        "n_blocks": 1,
        "found": 1,
    }

    _dump(
        debug_dir=tmp_path,
        name="sample.jpg",
        image=image,
        blocks=blocks,
        fields=fields,
        status=status,
        info=info,
    )

    json_file = tmp_path / "sample.json"
    overlay_file = tmp_path / "sample_overlay.jpg"

    assert json_file.exists()
    assert overlay_file.exists()
