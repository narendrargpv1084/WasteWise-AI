from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

from config import MODEL_PATH
from ml.loader import get_class_names
from ml.predict import predict_image


def test_class_names_contains_twelve_classes():
    names = get_class_names()
    assert isinstance(names, list)
    assert len(names) == 12
    assert len(set(names)) == 12


@pytest.mark.skipif(not Path(MODEL_PATH).is_file(), reason="Trained model file is not present")
def test_prediction_maps_to_known_class():
    image = Image.new("RGB", (224, 224), color=(180, 180, 180))
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    buffer.seek(0)

    result = predict_image(buffer)
    names = get_class_names()

    assert result["class"] in names
    assert result["index"] in range(len(names))
    assert names[result["index"]] == result["class"]
