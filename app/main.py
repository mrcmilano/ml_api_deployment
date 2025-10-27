from __future__ import annotations
import logging
import os
import threading
import time
from pathlib import Path
from typing import Awaitable, Callable, Optional
import joblib
from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel
from sklearn.preprocessing import LabelEncoder
from app.utils import ProbabilisticTextClassifier, predict_language_safe

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
CONFIDENCE_THRESHOLD: float = 0.5
MODEL_VERSION = os.getenv("MODEL_VERSION", "v1")
DEFAULT_MODEL_DIR = Path("models") / "text" / "language_classification" / MODEL_VERSION
MODEL_DIR = Path(os.getenv("MODEL_DIR", str(DEFAULT_MODEL_DIR)))
MODEL_FILENAME = "best_classifier.pkl"
LABEL_ENCOD_FILENAME = "label_encoder.pkl"
SKIP_MODEL_LOADING = os.getenv("SKIP_MODEL_LOADING", "false").lower() in {"1", "true", "yes"}

APP_ENV = os.getenv("APP_ENV", "dev").lower()

model_lock = threading.Lock()

def load_artifacts() -> tuple[Optional[ProbabilisticTextClassifier], Optional[LabelEncoder]]:
    """Carica modelli e altri artefatti necessari."""
    try:
        if not MODEL_DIR.exists():
            raise FileNotFoundError(f"Model directory {MODEL_DIR} does not exist.")
        if not (MODEL_DIR / MODEL_FILENAME).is_file():
            raise FileNotFoundError(f"Model file {MODEL_FILENAME} not found in {MODEL_DIR}.")
        if not (MODEL_DIR / LABEL_ENCOD_FILENAME).is_file():
            raise FileNotFoundError(f"Label encoder file {LABEL_ENCOD_FILENAME} not found in {MODEL_DIR}.")
    except FileNotFoundError as exc:
        if APP_ENV == "prod":
            logger.warning("Skipping model load in prod environment: %s", exc)
            return None, None
        raise

    model: ProbabilisticTextClassifier = joblib.load(MODEL_DIR / MODEL_FILENAME)
    label_encoder: LabelEncoder = joblib.load(MODEL_DIR / LABEL_ENCOD_FILENAME)
    return model, label_encoder

# -------------------------------
# Load model and encoder
# -------------------------------
model: Optional[ProbabilisticTextClassifier] = None
le: Optional[LabelEncoder] = None

if SKIP_MODEL_LOADING:
    logger.info("Skipping model load because SKIP_MODEL_LOADING is set.")
else:
    model, le = load_artifacts()
    if model is None or le is None:
        logger.info("Model artifacts not loaded; predictions disabled for this run.")
    else:
        logger.info("Loaded model version %s from %s", MODEL_VERSION, MODEL_DIR)

# -------------------------------
# FastAPI setup
# -------------------------------
app = FastAPI(title="Language Detection API")

# Middleware per logging delle richieste
@app.middleware("http")
async def log_requests(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    start_time = time.time()
    # Log info di inizio
    logger.info("Request start: %s %s", request.method, request.url)
    try:
        response = await call_next(request)
    except Exception as exc:
        # Log dell’errore
        logger.error("Exception in request %s %s: %s", request.method, request.url, exc, exc_info=True)
        raise
    process_time = time.time() - start_time
    logger.info(
        "Request end: %s %s completed_in=%.4fs status_code=%s",
        request.method,
        request.url,
        process_time,
        response.status_code,
    )
    return response

# -------------------------------
# Input schema
# -------------------------------
class TextsRequest(BaseModel):
    texts: list[str]


class LanguageDetectionResponse(BaseModel):
    predictions: list[str]
    model_version: str

# -------------------------------
# Health check endpoint
# -------------------------------
@app.get("/status")
def status() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/language_detection", response_model=LanguageDetectionResponse)
def detect_language(request: TextsRequest) -> LanguageDetectionResponse:
    texts: list[str] = request.texts
    
    if not texts:  # controllo lista vuota
        logger.warning("Prediction request received with empty text list")
        return LanguageDetectionResponse(predictions=[], model_version=MODEL_VERSION)

    if model is None or le is None:
        logger.error("Prediction attempt without loaded model artifacts")
        raise HTTPException(status_code=503, detail="Model artifacts not loaded")

    with model_lock:
        predictions = predict_language_safe(model, le, texts, threshold=CONFIDENCE_THRESHOLD)
        logger.info("Predictions made: model_version=%s, n_texts=%s", MODEL_VERSION, len(texts))
    
    return LanguageDetectionResponse(predictions=predictions, model_version=MODEL_VERSION)


# -------------------------------
# Model version endpoint
# -------------------------------
@app.get("/model_version")
def get_model_version() -> dict[str, str]:
    return {"model_version": MODEL_VERSION}
