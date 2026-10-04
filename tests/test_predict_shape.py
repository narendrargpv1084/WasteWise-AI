"""Regression tests for the TFLite prediction pipeline.

Verifies:
- The TFLite interpreter loads successfully.
- Input shape and dtype are correct.
- A sample image produces a valid class name and index.
- No TensorFlow/Keras model is loaded by the production inference path.
"""
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

from config import MODEL_PATH
from ml.loader import get_class_names, get_interpreter


def _make_image_buffer(width=224, height=224, color=(128, 200, 80)):
    img = Image.new("RGB", (width, height), color=color)
    buf = BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf


# ---------------------------------------------------------------------------
# Class-names sanity checks
# ---------------------------------------------------------------------------


def test_class_names_contains_twelve_classes():
    names = get_class_names()
    assert isinstance(names, list)
    assert len(names) == 12
    assert len(set(names)) == 12


def test_class_names_are_expected_values():
    expected = {
        "battery", "biological", "brown-glass", "cardboard", "clothes",
        "green-glass", "metal", "paper", "plastic", "shoes", "trash", "white-glass",
    }
    assert set(get_class_names()) == expected


# ---------------------------------------------------------------------------
# TFLite interpreter checks
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not Path(MODEL_PATH).is_file(),
    reason="TFLite model file is not present",
)
def test_interpreter_input_shape():
    interpreter = get_interpreter()
    details = interpreter.get_input_details()[0]
    assert tuple(details["shape"].tolist()) == (1, 224, 224, 3)


@pytest.mark.skipif(
    not Path(MODEL_PATH).is_file(),
    reason="TFLite model file is not present",
)
def test_interpreter_input_dtype():
    interpreter = get_interpreter()
    details = interpreter.get_input_details()[0]
    assert details["dtype"] == np.float32


@pytest.mark.skipif(
    not Path(MODEL_PATH).is_file(),
    reason="TFLite model file is not present",
)
def test_interpreter_output_shape():
    interpreter = get_interpreter()
    details = interpreter.get_output_details()[0]
    # Output should be [1, 12] — one row, 12 class scores.
    assert tuple(details["shape"].tolist()) == (1, 12)


# ---------------------------------------------------------------------------
# End-to-end prediction checks
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not Path(MODEL_PATH).is_file(),
    reason="TFLite model file is not present",
)
def test_prediction_maps_to_known_class():
    from ml.predict import predict_image

    result = predict_image(_make_image_buffer())
    names = get_class_names()

    assert result["class"] in names
    assert 0 <= result["index"] <= 11
    assert names[result["index"]] == result["class"]
    assert "confidence" in result
    assert "top_k" in result


@pytest.mark.skipif(
    not Path(MODEL_PATH).is_file(),
    reason="TFLite model file is not present",
)
def test_prediction_top_k_structure():
    from ml.predict import predict_image

    result = predict_image(_make_image_buffer(), top_k=3)
    names = get_class_names()

    assert len(result["top_k"]) <= 3
    for entry in result["top_k"]:
        assert entry["class"] in names
        assert "confidence" in entry


@pytest.mark.skipif(
    not Path(MODEL_PATH).is_file(),
    reason="TFLite model file is not present",
)
def test_multiple_images_predict_valid_classes():
    """Run predictions on several synthetic images with different colours."""
    from ml.predict import predict_image

    colours = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (180, 180, 180),
        (30, 30, 30),
    ]
    names = get_class_names()
    for colour in colours:
        result = predict_image(_make_image_buffer(color=colour))
        assert result["class"] in names
        assert 0 <= result["index"] <= 11


# ---------------------------------------------------------------------------
# Safety: production inference must NOT import tensorflow.keras
# ---------------------------------------------------------------------------


def test_production_path_does_not_import_keras(monkeypatch):
    """Ensure get_interpreter() never calls tensorflow.keras.models.load_model."""
    import importlib
    import sys

    # Clear the cached interpreter so loader code runs fresh.
    from ml import loader
    loader.get_interpreter.cache_clear()

    # Patch tensorflow so an import attempt raises immediately.
    sentinel = object()

    class _FakeTF:
        keras = sentinel

    monkeypatch.setitem(sys.modules, "tensorflow", _FakeTF())

    # Should still work — ai_edge_litert does not depend on tensorflow.
    if Path(MODEL_PATH).is_file():
        interp = get_interpreter()
        assert interp is not None

    # Restore cache state for subsequent tests.
    loader.get_interpreter.cache_clear()
