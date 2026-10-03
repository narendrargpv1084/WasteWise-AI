from io import BytesIO

import numpy as np
from PIL import Image

from ml.preprocess import preprocess_image


def test_preprocess_output_shape():
    image = Image.new("RGB", (640, 480), color=(34, 139, 34))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)

    batch = preprocess_image(buffer)

    assert batch.shape == (1, 224, 224, 3)
    assert batch.dtype == np.float32
    assert batch.min() >= 0.0
    assert batch.max() <= 255.0
