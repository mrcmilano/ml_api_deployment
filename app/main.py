import logging
import os
import threading
import time
from typing import List

import joblib
from fastapi import FastAPI, Request
from pydantic import BaseModel

# clean_texts usata da predict_language_safe nella pipeline
from app.utils import predict_language_safe, clean_texts

# -------------------------------
# Logging setup
# -------------------------------
# TODO: aggiungere logging su file con rotazione
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger("app")  # logger dell’applicazione

# -------------------------------
# Config
# -------------------------------
CONFIDENCE_THRESHOLD = 0.5
MODEL_VERSION = "v1"
MODEL_DIR = f"models/text/language_classification/{MODEL_VERSION}"
MODEL_FILENAME = "best_classifier.pkl"
LABEL_ENCOD_FILENAME = "label_encoder.pkl"
SKIP_MODEL_LOADING = os.getenv("SKIP_MODEL_LOADING", "false").lower() in {"1", "true", "yes"}

model_lock = threading.Lock()


def load_artifacts():
    """Carica modelli e altri artefatti necessari"""
    if not os.path.exists(MODEL_DIR):
        raise FileNotFoundError(f"Model directory {MODEL_DIR} does not exist.")
    if not os.path.isfile(os.path.join(MODEL_DIR, MODEL_FILENAME)):
        raise FileNotFoundError(f"Model file {MODEL_FILENAME} not found in {MODEL_DIR}.")
    if not os.path.isfile(os.path.join(MODEL_DIR, LABEL_ENCOD_FILENAME)):
        raise FileNotFoundError(f"Label encoder file {LABEL_ENCOD_FILENAME} not found in {MODEL_DIR}.")
# Load model and encoder on module import unless explicitly skipped
model = None
le = None

if SKIP_MODEL_LOADING:
    logger.info("Skipping model load because SKIP_MODEL_LOADING is set.")
else:
    model, le = load_artifacts()
    logger.info(f"Loaded model version {MODEL_VERSION} from {MODEL_DIR}")

# -------------------------------
# FastAPI setup
# -------------------------------
app = FastAPI(title="Language Detection API")

# Middleware per logging delle richieste
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    # Log info di inizio
    logger.info(f"Request start: {request.method} {request.url}")
    try:
        response = await call_next(request)
    except Exception as exc:
        # Log dell’errore
        logger.error(f"Exception in request {request.method} {request.url}: {exc}", exc_info=True)
        raise
    process_time = time.time() - start_time
    logger.info(f"Request end: {request.method} {request.url} completed_in={process_time:.4f}s status_code={response.status_code}")
    return response

# -------------------------------
# Input schema
# -------------------------------
class TextsRequest(BaseModel):
    texts: List[str]

# -------------------------------
# Health check endpoint
# -------------------------------
@app.get("/status")
def status():
    return {"status": "ok"}


@app.post("/language_detection")
def detect_language(request: TextsRequest):
    texts = request.texts
    
    if not texts:  # controllo lista vuota
        logger.warning("Prediction request received with empty text list")
        return {"predictions": [], "model_version": MODEL_VERSION}

    with model_lock:
        predictions = predict_language_safe(model, le, texts, threshold=CONFIDENCE_THRESHOLD)
        logger.info(f"Predictions made: model_version={MODEL_VERSION}, n_texts={len(texts)}")
    
    return {"predictions": predictions, "model_version": MODEL_VERSION}


# -------------------------------
# Model version endpoint
# -------------------------------
@app.get("/model_version")
def get_model_version():
    return {"model_version": MODEL_VERSION}
