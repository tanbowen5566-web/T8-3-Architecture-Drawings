from unittest.mock import patch
from PIL import Image

from src.no_tb.branch import process_no_tb


def test_process_no_tb_rules_branch(tmp_path):
    image_path = tmp_path / "test.jpg"
    Image.new("RGB", (100, 80)).save(image_path)

    expected = ({"drawing_name": {"value": "PLAN", "confidence": 0.9}}, {"mode": "rules"})

    with patch("src.no_tb.branch.run_rules", return_value=expected) as mock_rules:
        result = process_no_tb(
            image_path=image_path,
            ocr_model=None,
            processor=None,
            field_model=None,
            model_fields=["drawing_name"],
            method="rules",
            target_long=4096,
        )

    assert result == expected
    mock_rules.assert_called_once()


def test_process_no_tb_layoutlm_branch(tmp_path):
    image_path = tmp_path / "test.jpg"
    Image.new("RGB", (100, 80)).save(image_path)

    expected = ({"drawing_name": {"value": "SECTION", "confidence": 0.8}}, {"mode": "layoutlm"})

    with patch("src.no_tb.branch.run_baseline", return_value=expected) as mock_baseline:
        result = process_no_tb(
            image_path=image_path,
            ocr_model=None,
            processor=None,
            field_model=None,
            model_fields=["drawing_name"],
            method="layoutlm",
        )

    assert result == expected
    mock_baseline.assert_called_once()
