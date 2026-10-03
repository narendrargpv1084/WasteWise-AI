# WasteWise AI

Flask web application for classifying household waste images with a trained MobileNetV2 model. The model is not retrained in this repository.

## Model

- File: `model/wastewise_best.keras`
- Architecture: MobileNetV2 transfer learning
- Input: RGB image, 224x224
- Classes: 12 labels in `model/class_names.json`
- Validation accuracy: approximately 95%
- The Keras graph already includes `Rescaling(1./127.5, offset=-1)`. Application code only converts to RGB, resizes, and batches.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

On macOS or Linux, activate with `source .venv/bin/activate` and copy the env file with `cp .env.example .env`. Set `SECRET_KEY` in `.env` before any public deployment.

Confirm these files exist before starting the server:

- `model/wastewise_best.keras`
- `model/class_names.json`
- `model/model_info.json`

## Run locally

```bash
python app.py
```

Then open `http://127.0.0.1:5000`.

## Production

Windows (Waitress):

```bash
waitress-serve --listen=0.0.0.0:8000 wsgi:app
```

Linux (Gunicorn):

```bash
gunicorn --bind 0.0.0.0:8000 wsgi:app
```

## HTTP routes

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/` | Upload and camera capture |
| GET | `/about` | Model and workflow notes |
| POST | `/predict` | HTML classification result |
| POST | `/api/predict` | JSON classification (`multipart/form-data`, field `image`) |

Public API success body:

```json
{
  "success": true,
  "class": "plastic"
}
```

Confidence is computed internally and is not returned to users.

## Tests

```bash
pytest
```

Tests check preprocess shape `(1, 224, 224, 3)`, that there are 12 class names, and that a prediction maps to one of those classes when the trained model file is present.
