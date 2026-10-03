import base64
import logging
import os
import uuid
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename

from config import (
    ALLOWED_EXTENSIONS,
    Config,
    LOG_DIR,
    UPLOAD_DIR,
)
from ml.predict import predict_image

_startup_logger = logging.getLogger(__name__)
_model_initialized = False


def _initialize_model() -> None:
    """Load and warm up the model at startup — once per process.

    Two guards are applied:
    1. A module-level flag ensures only the first call does real work, even
       if create_app() is called multiple times in the same process.
    2. The Flask development reloader sets WERKZEUG_RUN_MAIN=true only in
       the child process that actually serves requests.  When debug=True the
       parent (monitor) process skips initialization; the child runs it once.
       When debug=False (production/wsgi) WERKZEUG_RUN_MAIN is absent and we
       always initialize — which is the desired behavior.
    """
    global _model_initialized
    if _model_initialized:
        return

    werkzeug_main = os.environ.get("WERKZEUG_RUN_MAIN")
    # debug=False → werkzeug_main is None → always initialize (correct).
    # debug=True  → only initialize inside the reloader child (werkzeug_main == "true").
    if werkzeug_main is not None and werkzeug_main != "true":
        return

    _model_initialized = True
    try:
        from ml.loader import warmup_model
        warmup_model()
    except Exception:
        _model_initialized = False  # allow retry if startup is retried
        _startup_logger.exception(
            "WasteWise AI model failed to load at startup. "
            "Predictions will be unavailable until the issue is resolved."
        )
        raise

DISPOSAL_GUIDANCE = {
    "battery": {
        "title": "Take batteries to a hazardous-waste drop-off",
        "summary": "Household batteries contain metals that should not go in regular trash or recycling bins.",
        "steps": [
            "Tape the terminals on lithium and button batteries.",
            "Store used batteries in a dry container away from heat.",
            "Take them to a battery retailer, municipal hazardous-waste site, or e-waste collection event.",
        ],
    },
    "biological": {
        "title": "Compost or use the organics bin",
        "summary": "Food scraps and other biological waste break down best in compost, not landfill.",
        "steps": [
            "Remove packaging, stickers, and non-food items.",
            "Place scraps in a compost or food-waste bin if your city provides one.",
            "Avoid liquids and oils that can contaminate collection.",
        ],
    },
    "brown-glass": {
        "title": "Recycle brown glass with color-sorted glass",
        "summary": "Amber bottles are recyclable when clean and kept with other brown glass.",
        "steps": [
            "Empty and rinse the container.",
            "Remove lids and non-glass attachments when possible.",
            "Place in the glass recycling stream specified for brown or mixed glass locally.",
        ],
    },
    "cardboard": {
        "title": "Flatten and recycle cardboard",
        "summary": "Clean, dry cardboard is widely accepted in paper recycling.",
        "steps": [
            "Remove tape, plastic windows, and packing fillers.",
            "Flatten boxes to save space.",
            "Keep cardboard dry; greasy pizza boxes belong in compost or trash depending on local rules.",
        ],
    },
    "clothes": {
        "title": "Reuse, donate, or use textile recycling",
        "summary": "Wearable clothing should stay in use. Worn textiles can often be recycled separately.",
        "steps": [
            "Donate clean, wearable items to a charity or reuse shop.",
            "Use a textile recycling bin for damaged clothing.",
            "Do not put textiles in mixed recycling unless your program allows it.",
        ],
    },
    "green-glass": {
        "title": "Recycle green glass with color-sorted glass",
        "summary": "Green bottles are recyclable when emptied and kept with other green glass.",
        "steps": [
            "Empty and rinse the bottle.",
            "Remove corks, caps, and metal or plastic collars.",
            "Place in the glass recycling stream specified for green or mixed glass locally.",
        ],
    },
    "metal": {
        "title": "Rinse and recycle metal",
        "summary": "Aluminum and steel containers are high-value recyclables when clean.",
        "steps": [
            "Empty and rinse cans or tins.",
            "Let them dry to reduce contamination.",
            "Place in the metal or mixed-recycling bin according to local rules.",
        ],
    },
    "paper": {
        "title": "Recycle clean paper",
        "summary": "Dry paper belongs in paper recycling; wet or food-soiled paper usually does not.",
        "steps": [
            "Keep paper clean and dry.",
            "Remove plastic windows, bindings, and sticky residue when practical.",
            "Shredded paper may need a paper bag; check local guidance.",
        ],
    },
    "plastic": {
        "title": "Check the resin type, then recycle if accepted",
        "summary": "Not all plastics are equal. Local programs often accept bottles and jugs more readily than film.",
        "steps": [
            "Empty, rinse, and recap only if your program asks you to.",
            "Look for bottles, tubs, and jugs commonly marked #1 or #2.",
            "Keep plastic bags and film out of curbside bins unless a store drop-off exists.",
        ],
    },
    "shoes": {
        "title": "Donate wearable shoes or use a shoe take-back",
        "summary": "Paired, wearable shoes can be reused. Damaged pairs often need specialty recycling.",
        "steps": [
            "Donate clean, wearable pairs to a reuse organization.",
            "Use brand or retailer take-back programs for worn-out shoes when available.",
            "Do not place shoes in mixed recycling.",
        ],
    },
    "trash": {
        "title": "Place in general waste",
        "summary": "Items that cannot be reused, composted, or recycled belong in residual waste.",
        "steps": [
            "Confirm the item is not hazardous (batteries, chemicals, e-waste).",
            "Bag residual waste according to local collection rules.",
            "Consider whether a similar item could be avoided, repaired, or bought in a recyclable form next time.",
        ],
    },
    "white-glass": {
        "title": "Recycle clear glass with color-sorted glass",
        "summary": "Clear bottles and jars recycle well when emptied and kept with other clear glass.",
        "steps": [
            "Empty and rinse the container.",
            "Remove lids, pumps, and non-glass parts.",
            "Place in the glass recycling stream specified for clear or mixed glass locally.",
        ],
    },
}

DISPLAY_NAMES = {
    "battery": "Battery",
    "biological": "Biological / food waste",
    "brown-glass": "Brown glass",
    "cardboard": "Cardboard",
    "clothes": "Clothes",
    "green-glass": "Green glass",
    "metal": "Metal",
    "paper": "Paper",
    "plastic": "Plastic",
    "shoes": "Shoes",
    "trash": "General trash",
    "white-glass": "Clear glass",
}


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)
    app.json.sort_keys = False

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    _initialize_model()

    def allowed_file(filename):
        return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

    def save_upload(file_storage):
        original = secure_filename(file_storage.filename or "")
        if not original or not allowed_file(original):
            return None, "Please upload a PNG, JPG, JPEG, or WEBP image."

        extension = original.rsplit(".", 1)[1].lower()
        stored_name = f"{uuid.uuid4().hex}.{extension}"
        destination = UPLOAD_DIR / stored_name
        # Prevent path traversal: destination must stay inside UPLOAD_DIR.
        if destination.resolve().parent != UPLOAD_DIR.resolve():
            return None, "Invalid upload path."

        file_storage.save(destination)
        return destination, None

    def encode_image(path):
        suffix = path.suffix.lower().lstrip(".")
        mime = "jpeg" if suffix in {"jpg", "jpeg"} else suffix
        data = path.read_bytes()
        encoded = base64.b64encode(data).decode("ascii")
        return f"data:image/{mime};base64,{encoded}"

    def cleanup(path):
        try:
            if path and Path(path).is_file():
                Path(path).unlink()
        except OSError:
            app.logger.warning("Could not delete temporary upload %s", path)

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/about")
    def about():
        return render_template("about.html")

    @app.route("/predict", methods=["POST"])
    def predict():
        if "image" not in request.files:
            return render_template(
                "index.html",
                error="Choose an image before classifying.",
            ), 400

        file_storage = request.files["image"]
        if not file_storage or not file_storage.filename:
            return render_template(
                "index.html",
                error="Choose an image before classifying.",
            ), 400

        saved_path, error = save_upload(file_storage)
        if error:
            return render_template("index.html", error=error), 400

        try:
            result = predict_image(saved_path)
            predicted = result["class"]
            preview = encode_image(saved_path)
        except Exception:
            app.logger.exception("Prediction failed")
            return render_template(
                "index.html",
                error="Classification failed. Please try another image.",
            ), 500
        finally:
            cleanup(saved_path)

        guidance = DISPOSAL_GUIDANCE.get(predicted, DISPOSAL_GUIDANCE["trash"])
        return render_template(
            "result.html",
            predicted_class=predicted,
            display_name=DISPLAY_NAMES.get(predicted, predicted.replace("-", " ").title()),
            guidance=guidance,
            image_data=preview,
        )

    @app.route("/api/predict", methods=["POST"])
    def api_predict():
        if "image" not in request.files:
            return jsonify({"success": False, "error": "Missing image file."}), 400

        file_storage = request.files["image"]
        if not file_storage or not file_storage.filename:
            return jsonify({"success": False, "error": "Missing image file."}), 400

        saved_path, error = save_upload(file_storage)
        if error:
            return jsonify({"success": False, "error": error}), 400

        try:
            result = predict_image(saved_path)
            return jsonify({"success": True, "class": result["class"]})
        except Exception:
            app.logger.exception("API prediction failed")
            return jsonify(
                {"success": False, "error": "Classification failed. Please try another image."}
            ), 500
        finally:
            cleanup(saved_path)

    @app.errorhandler(404)
    def not_found(_error):
        if request.path.startswith("/api/"):
            return jsonify({"success": False, "error": "Not found."}), 404
        return render_template("errors/404.html"), 404

    @app.errorhandler(RequestEntityTooLarge)
    def too_large(_error):
        if request.path.startswith("/api/"):
            return jsonify({"success": False, "error": "File too large."}), 413
        return render_template("errors/413.html"), 413

    @app.errorhandler(413)
    def payload_too_large(_error):
        if request.path.startswith("/api/"):
            return jsonify({"success": False, "error": "File too large."}), 413
        return render_template("errors/413.html"), 413

    @app.errorhandler(500)
    def server_error(_error):
        if request.path.startswith("/api/"):
            return jsonify({"success": False, "error": "Something went wrong."}), 500
        return render_template(
            "index.html",
            error="Something went wrong. Please try again.",
        ), 500

    return app


app = create_app()

if __name__ == "__main__":
    app.run(debug=False)
