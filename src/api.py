"""FastAPI model-serving application with structured logging and Prometheus metrics.

Run locally: uvicorn src.api:app --reload
Docs:        http://localhost:8000/docs
"""
import json
import logging
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest,
)
from pydantic import BaseModel, Field

from src.config import FEATURE_COLUMNS, MODEL_INFO_PATH, MODEL_PATH

# ------------------------------------------------------------------ logging
logger = logging.getLogger("heart-api")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    if os.getenv("LOG_FILE"):
        fh = logging.FileHandler(os.environ["LOG_FILE"])
        fh.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(fh)
    logger.setLevel(logging.INFO)
    logger.propagate = False


def log_event(event: str, **fields):
    """Emit one JSON log line (easy to ship to ELK/Loki/CloudWatch)."""
    logger.info(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                            "event": event, **fields}, default=str))


# ------------------------------------------------------------------ metrics
REQUESTS = Counter("http_requests_total", "HTTP requests", ["method", "path", "status"])
LATENCY = Histogram("http_request_duration_seconds", "Request latency (s)", ["path"],
                    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5))
PREDICTIONS = Counter("predictions_total", "Predictions by class", ["label"])
CONFIDENCE = Histogram("prediction_confidence", "Confidence of predicted class",
                       buckets=(0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0))
PROB_DISEASE = Histogram("prediction_probability_disease", "P(heart disease)",
                         buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0))
MODEL_LOADED = Gauge("model_loaded", "1 if the model is loaded")
IN_PROGRESS = Gauge("http_requests_in_progress", "Requests currently being served")

STATE = {"model": None, "info": {}}


@asynccontextmanager
async def lifespan(app: FastAPI):
    path = os.getenv("MODEL_PATH", str(MODEL_PATH))
    try:
        STATE["model"] = joblib.load(path)
        info_path = os.getenv("MODEL_INFO_PATH", str(MODEL_INFO_PATH))
        if os.path.exists(info_path):
            with open(info_path) as fh:
                STATE["info"] = json.load(fh)
        MODEL_LOADED.set(1)
        log_event("model_loaded", path=path, version=STATE["info"].get("version", "unknown"))
    except Exception as exc:
        STATE["model"] = None
        MODEL_LOADED.set(0)
        log_event("model_load_failed", path=path, error=str(exc))
    yield
    log_event("shutdown")


app = FastAPI(
    title="Heart Disease Risk API",
    description="Predicts the probability of heart disease from patient health data.",
    version="1.0.0",
    lifespan=lifespan,
)


# ------------------------------------------------------------------ schemas
class PatientFeatures(BaseModel):
    age: float = Field(..., ge=1, le=120, description="Age in years")
    sex: int = Field(..., ge=0, le=1, description="1 = male, 0 = female")
    cp: int = Field(..., ge=1, le=4, description="Chest pain type (1-4)")
    trestbps: float = Field(..., ge=50, le=300, description="Resting BP (mm Hg)")
    chol: float = Field(..., ge=50, le=700, description="Serum cholesterol (mg/dl)")
    fbs: int = Field(..., ge=0, le=1, description="Fasting blood sugar > 120 mg/dl")
    restecg: int = Field(..., ge=0, le=2, description="Resting ECG result")
    thalach: float = Field(..., ge=40, le=250, description="Max heart rate achieved")
    exang: int = Field(..., ge=0, le=1, description="Exercise-induced angina")
    oldpeak: float = Field(..., ge=0, le=10, description="ST depression")
    slope: int = Field(..., ge=1, le=3, description="Slope of peak exercise ST segment")
    ca: Optional[float] = Field(None, ge=0, le=4, description="Major vessels (optional)")
    thal: Optional[float] = Field(None, ge=3, le=7, description="Thalassemia (optional)")

    model_config = {"json_schema_extra": {"example": {
        "age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233, "fbs": 1,
        "restecg": 2, "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 3,
        "ca": 0, "thal": 6}}}


class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="1 = heart disease likely, 0 = unlikely")
    label: str
    confidence: float = Field(..., description="Probability of the predicted class")
    probability_disease: float
    risk_level: str
    model_version: str
    request_id: str


# --------------------------------------------------------------- middleware
@app.middleware("http")
async def observe(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", uuid.uuid4().hex[:12])
    request.state.request_id = request_id
    start = time.perf_counter()
    IN_PROGRESS.inc()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        response.headers["X-Request-ID"] = request_id
        return response
    finally:
        IN_PROGRESS.dec()
        elapsed = time.perf_counter() - start
        route = request.scope.get("route")
        path = route.path if route else "unmatched"
        REQUESTS.labels(request.method, path, str(status)).inc()
        LATENCY.labels(path).observe(elapsed)
        if path != "/metrics":
            log_event("request", request_id=request_id, method=request.method,
                      path=request.url.path, status=status,
                      latency_ms=round(elapsed * 1000, 2),
                      client=request.client.host if request.client else None)


# ---------------------------------------------------------------- endpoints
@app.get("/")
def root():
    return {"service": "heart-disease-risk-api", "docs": "/docs", "predict": "POST /predict"}


@app.get("/health")
def health():
    if STATE["model"] is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return {"status": "ok", "model_loaded": True}


@app.get("/model-info")
def model_info():
    info = STATE["info"]
    return {k: info.get(k) for k in ("model_name", "version", "trained_at",
                                     "selection_metric", "sklearn_version", "features")}


@app.post("/predict", response_model=PredictionResponse)
def predict(patient: PatientFeatures, request: Request):
    model = STATE["model"]
    if model is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    frame = pd.DataFrame([patient.model_dump()])[FEATURE_COLUMNS].astype("float64")
    try:
        p_disease = float(model.predict_proba(frame)[0][1])
    except Exception as exc:
        log_event("prediction_error", request_id=request.state.request_id, error=str(exc))
        raise HTTPException(status_code=500, detail="prediction failed")

    pred = int(p_disease >= 0.5)
    confidence = p_disease if pred == 1 else 1.0 - p_disease
    risk = "low" if p_disease < 0.33 else "moderate" if p_disease < 0.66 else "high"
    label = "heart disease likely" if pred else "heart disease unlikely"

    PREDICTIONS.labels(str(pred)).inc()
    CONFIDENCE.observe(confidence)
    PROB_DISEASE.observe(p_disease)
    log_event("prediction", request_id=request.state.request_id, prediction=pred,
              probability_disease=round(p_disease, 4), risk_level=risk)

    return PredictionResponse(
        prediction=pred, label=label, confidence=round(confidence, 4),
        probability_disease=round(p_disease, 4), risk_level=risk,
        model_version=str(STATE["info"].get("version", "unknown")),
        request_id=request.state.request_id)


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
