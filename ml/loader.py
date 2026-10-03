import json
import logging
from functools import lru_cache

from config import CLASS_NAMES_PATH, MODEL_PATH

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_model():
    """Load the trained Keras model once. Do not compile or train."""
    from tensorflow.keras.models import load_model

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    return load_model(MODEL_PATH, compile=False)


@lru_cache(maxsize=1)
def get_class_names():
    """Load class-index mapping once from class_names.json."""
    if not CLASS_NAMES_PATH.is_file():
        raise FileNotFoundError(f"Class names file not found: {CLASS_NAMES_PATH}")
    with open(CLASS_NAMES_PATH, encoding="utf-8") as handle:
        names = json.load(handle)
    if not isinstance(names, list) or not names:
        raise ValueError("class_names.json must be a non-empty JSON list")
    return names


def warmup_model():
    """Load and warm up the model with one dummy inference.

    Uses get_model() so the lru_cache(maxsize=1) instance is reused by all
    subsequent calls.  Safe to call more than once — the model is only loaded
    and inferred once thanks to the cache.
    """
    import numpy as np

    logger.info("Loading WasteWise AI model...")
    model = get_model()

    logger.info("Warming up WasteWise AI model...")
    dummy_input = np.zeros((1, 224, 224, 3), dtype=np.float32)
    model.predict(dummy_input, verbose=0)

    logger.info("WasteWise AI model ready.")
    return model
