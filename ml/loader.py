import json
import logging
from functools import lru_cache

from config import CLASS_NAMES_PATH, MODEL_PATH

logger = logging.getLogger(__name__)

_EXPECTED_INPUT_SHAPE = (1, 224, 224, 3)
_EXPECTED_INPUT_DTYPE = "float32"


@lru_cache(maxsize=1)
def get_interpreter():
    """Load the TFLite model as an Interpreter once and cache it.

    Uses ai-edge-litert (Google's maintained TFLite runtime).
    Validates input shape and dtype against the expected values.
    """
    from ai_edge_litert.interpreter import Interpreter

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"TFLite model file not found: {MODEL_PATH}")

    interpreter = Interpreter(model_path=str(MODEL_PATH))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    actual_shape = tuple(input_details["shape"].tolist())
    actual_dtype = input_details["dtype"].__name__

    if actual_shape != _EXPECTED_INPUT_SHAPE:
        raise ValueError(
            f"TFLite model input shape mismatch: expected {_EXPECTED_INPUT_SHAPE}, "
            f"got {actual_shape}"
        )
    if actual_dtype != _EXPECTED_INPUT_DTYPE:
        raise ValueError(
            f"TFLite model input dtype mismatch: expected {_EXPECTED_INPUT_DTYPE}, "
            f"got {actual_dtype}"
        )

    return interpreter


# Keep get_model as an alias so any caller that still uses get_model() keeps working.
get_model = get_interpreter


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
    """Load and warm up the TFLite interpreter with one dummy inference.

    Uses get_interpreter() so the lru_cache(maxsize=1) instance is reused by
    all subsequent calls.  Safe to call more than once — the model is only
    loaded and inferred once thanks to the cache.
    """
    import numpy as np

    logger.info("Loading WasteWise AI TFLite model...")
    interpreter = get_interpreter()

    logger.info("Warming up WasteWise AI TFLite model...")
    input_details = interpreter.get_input_details()[0]
    dummy_input = np.zeros(_EXPECTED_INPUT_SHAPE, dtype=np.float32)
    interpreter.set_tensor(input_details["index"], dummy_input)
    interpreter.invoke()

    logger.info("WasteWise AI TFLite model ready.")
    return interpreter
