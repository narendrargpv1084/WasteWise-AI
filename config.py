import os
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

UPLOAD_DIR = _resolve(os.getenv("UPLOAD_DIR"), BASE_DIR / "uploads")
LOG_DIR = _resolve(os.getenv("LOG_DIR"), BASE_DIR / "logs")

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
MAX_CONTENT_LENGTH = int(os.getenv("MAX_CONTENT_LENGTH", 8 * 1024 * 1024))

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
