from pathlib import Path

import numpy as np
from PIL import Image

from config import INPUT_SIZE


def preprocess_image(image_source):
    """Convert an image to a model-ready batch without extra normalization.

    The saved Keras model already includes Rescaling(1./127.5, offset=-1).
    This function only:
      - opens the image
      - converts to RGB
      - resizes to 224x224
      - converts to a float32 numpy array in 0-255 range
      - adds a batch dimension
    """
    image = Image.open(image_source)
    image = image.convert("RGB")
    image = image.resize(INPUT_SIZE, Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32)
    return np.expand_dims(array, axis=0)


def preprocess_image_path(path):
    return preprocess_image(Path(path))
