import io
from PIL import Image

import pytest

from openoperator.perception.locator import TextLocatorEngine


def make_image_bytes(width=200, height=100, color=(255, 255, 255)):
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_find_text_targets_with_fuzzy_match(monkeypatch):
    # Mock pytesseract to return a single misspelled token 'notepadd'
    def fake_image_to_data(image, output_type=None):
        return {
            "text": ["notepadd"],
            "conf": ["95"],
            "left": [10],
            "top": [5],
            "width": [60],
            "height": [20],
            "block_num": [1],
            "par_num": [1],
            "line_num": [1],
        }

    import pytesseract
    monkeypatch.setattr(pytesseract, "image_to_data", fake_image_to_data)

    engine = TextLocatorEngine()
    image_bytes = make_image_bytes()
    results = engine.find_text_targets(image_bytes=image_bytes, search_text="notepad", fuzzy_threshold=0.7)

    assert results, "Expected fuzzy match to find a target for 'notepad' against 'notepadd'"
    assert any("notepadd" in t.text for t in results)


def test_find_text_targets_applies_crop(monkeypatch):
    # This fake will inspect the image.size to ensure crop applied
    captured = {}

    def fake_image_to_data(image, output_type=None):
        captured['size'] = image.size
        return {"text": [], "conf": [], "left": [], "top": [], "width": [], "height": [], "block_num": [], "par_num": [], "line_num": []}

    import pytesseract
    monkeypatch.setattr(pytesseract, "image_to_data", fake_image_to_data)

    engine = TextLocatorEngine()
    # original image 200x100
    image_bytes = make_image_bytes(200, 100)

    # Crop box: left=10, top=20, width=50, height=30
    crop_box = (10, 20, 50, 30)
    engine.find_text_targets(image_bytes=image_bytes, search_text="anything", crop_box=crop_box)

    # The image passed to pytesseract should have size (width, height)
    assert captured.get('size') == (50, 30)
