"""Developer validation script: batch inference over a folder of images.

Usage
-----
Run with pytest (skips if model missing):

    pytest tests/test_tflite_batch.py -v -s

Or directly as a script (accepts an optional folder argument):

    python tests/test_tflite_batch.py [path/to/image/folder]

The script will:
- Accept / use a test image folder.
- Run predictions on up to 50 images.
- Print: total images, successful predictions, failed predictions,
         predicted classes, and average inference time per image.

Note: confidence values are intentionally not printed — they are for
      internal use only and must not be exposed to end users.
"""

import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Allow running as a standalone script (python tests/test_tflite_batch.py)
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import MODEL_PATH  # noqa: E402
from ml.loader import get_class_names, get_interpreter  # noqa: E402
from ml.predict import predict_image  # noqa: E402

_SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
_MAX_IMAGES = 50


def run_batch(image_folder: Path) -> None:
    """Run predictions on up to _MAX_IMAGES images in *image_folder*."""
    images = sorted(
        p for p in image_folder.iterdir()
        if p.is_file() and p.suffix.lower() in _SUPPORTED_EXTS
    )[:_MAX_IMAGES]

    if not images:
        print(f"No supported images found in: {image_folder}")
        return

    class_names = get_class_names()
    total = len(images)
    successes = 0
    failures = 0
    predicted_classes: list[str] = []
    durations: list[float] = []

    print(f"\nWasteWise AI — TFLite batch validation")
    print(f"Model  : {MODEL_PATH}")
    print(f"Folder : {image_folder}")
    print(f"Images : {total}")
    print("-" * 60)

    for img_path in images:
        t_start = time.perf_counter()
        try:
            result = predict_image(img_path)
            t_end = time.perf_counter()
            successes += 1
            predicted_classes.append(result["class"])
            durations.append(t_end - t_start)
            print(f"  {img_path.name:<40} -> {result['class']}")
        except Exception as exc:
            t_end = time.perf_counter()
            failures += 1
            durations.append(t_end - t_start)
            print(f"  {img_path.name:<40} -> ERROR: {exc}")

    avg_ms = (sum(durations) / len(durations) * 1000) if durations else 0.0

    print("-" * 60)
    print(f"Total images          : {total}")
    print(f"Successful predictions: {successes}")
    print(f"Failed predictions    : {failures}")
    print(f"Average inference time: {avg_ms:.1f} ms/image")
    print(f"Classes predicted     : {sorted(set(predicted_classes))}")
    print()


# ---------------------------------------------------------------------------
# pytest integration
# ---------------------------------------------------------------------------

import pytest  # noqa: E402


@pytest.mark.skipif(
    not Path(MODEL_PATH).is_file(),
    reason="TFLite model file is not present",
)
def test_batch_synthetic_images(tmp_path):
    """Create 10 synthetic images and verify all are classified successfully."""
    from PIL import Image

    colours = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (200, 200, 200), (50, 100, 150),
        (10, 10, 10), (255, 255, 0), (0, 255, 255),
        (128, 0, 128), (255, 165, 0),
    ]
    for i, colour in enumerate(colours):
        img = Image.new("RGB", (224, 224), color=colour)
        img.save(tmp_path / f"synthetic_{i:02d}.jpg")

    images = sorted(tmp_path.glob("*.jpg"))
    class_names = get_class_names()
    failures = 0

    for img_path in images:
        try:
            result = predict_image(img_path)
            assert result["class"] in class_names
            assert 0 <= result["index"] <= 11
        except Exception:
            failures += 1

    assert failures == 0, f"{failures} out of {len(images)} synthetic images failed prediction"


# ---------------------------------------------------------------------------
# Script entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not Path(MODEL_PATH).is_file():
        print(f"ERROR: TFLite model not found at {MODEL_PATH}")
        sys.exit(1)

    folder_arg = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("tests/sample_images")

    if not folder_arg.is_dir():
        print(f"ERROR: image folder not found: {folder_arg}")
        print("Usage: python tests/test_tflite_batch.py [path/to/image/folder]")
        sys.exit(1)

    run_batch(folder_arg)
