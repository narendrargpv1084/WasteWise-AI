import numpy as np

from config import TOP_K
from ml.loader import get_class_names, get_interpreter
from ml.preprocess import preprocess_image


def predict_image(image_source, top_k=TOP_K):
    """Run TFLite inference and return a dictionary.

    Confidence and top-k are computed internally. Public routes should not
    expose confidence to end users.
    """
    batch = preprocess_image(image_source)

    # Ensure correct shape and dtype going into the interpreter.
    if batch.shape != (1, 224, 224, 3):
        raise ValueError(f"Unexpected batch shape: {batch.shape}; expected (1, 224, 224, 3)")
    if batch.dtype != np.float32:
        batch = batch.astype(np.float32)

    interpreter = get_interpreter()
    class_names = get_class_names()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    interpreter.set_tensor(input_details["index"], batch)
    interpreter.invoke()

    raw = interpreter.get_tensor(output_details["index"])
    probabilities = np.asarray(raw[0], dtype=np.float32)

    if probabilities.shape[0] != len(class_names):
        raise ValueError("Model output size does not match class_names.json")

    predicted_index = int(np.argmax(probabilities))
    predicted_class = class_names[predicted_index]
    confidence = float(probabilities[predicted_index])

    k = max(1, min(int(top_k), len(class_names)))
    top_indices = np.argsort(probabilities)[::-1][:k]
    top_predictions = [
        {
            "class": class_names[int(index)],
            "confidence": float(probabilities[int(index)]),
        }
        for index in top_indices
    ]

    return {
        "class": predicted_class,
        "index": predicted_index,
        "confidence": confidence,
        "top_k": top_predictions,
    }
