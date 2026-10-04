import os
import tempfile
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent


def _resolve(path_value, default):
    path = Path(path_value) if path_value else Path(default)
    if not path.is_absolute():
        path = BASE_DIR / path
    return path


MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = _resolve(os.getenv("MODEL_PATH"), MODEL_DIR / "wastewise_best.keras")
CLASS_NAMES_PATH = _resolve(os.getenv("CLASS_NAMES_PATH"), MODEL_DIR / "class_names.json")
MODEL_INFO_PATH = MODEL_DIR / "model_info.json"

# Use /tmp for uploads on Vercel (read-only filesystem except /tmp).
# Falls back to env override for local development flexibility.
_default_upload_dir = Path(tempfile.gettempdir()) / "wastewise_uploads"
UPLOAD_DIR = _resolve(os.getenv("UPLOAD_DIR"), _default_upload_dir)

# Use /tmp for logs on Vercel (deployed filesystem is read-only except /tmp).
# Falls back to env override for local development.
_default_log_dir = Path(tempfile.gettempdir()) / "wastewise_logs"
LOG_DIR = _resolve(os.getenv("LOG_DIR"), _default_log_dir)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
# 4 MB — safely below Vercel's ~4.5 MB request body hard limit.
MAX_CONTENT_LENGTH = int(os.getenv("MAX_CONTENT_LENGTH", 4 * 1024 * 1024))

SECRET_KEY = os.getenv("SECRET_KEY", "dev-change-me")
FLASK_ENV = os.getenv("FLASK_ENV", "production")

INPUT_SIZE = (224, 224)
TOP_K = 3


class Config:
    SECRET_KEY = SECRET_KEY
    MAX_CONTENT_LENGTH = MAX_CONTENT_LENGTH
    UPLOAD_FOLDER = str(UPLOAD_DIR)
    MODEL_PATH = str(MODEL_PATH)
    CLASS_NAMES_PATH = str(CLASS_NAMES_PATH)
