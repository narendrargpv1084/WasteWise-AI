import numpy as np

from config import TOP_K
from ml.loader import get_class_names, get_model
from ml.preprocess import preprocess_image


def predict_image(image_source, top_k=TOP_K):
    """Run inference and return a dictionary.

    Confidence and top-k are computed internally. Public routes should not
    expose confidence to end users.
    """
    batch = preprocess_image(image_source)
    model = get_model()
    class_names = get_class_names()

    raw = model.predict(batch, verbose=0)
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
