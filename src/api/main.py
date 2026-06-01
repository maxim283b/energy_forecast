import logging
import os
import subprocess
import sys
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from starlette.responses import Response

try:
    from prometheus_client import (
        CONTENT_TYPE_LATEST,
        Counter,
        Gauge,
        Histogram,
        generate_latest,
    )
except ImportError:  # pragma: no cover
    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"

    class _NoopMetric:
        def inc(self):
            return None

        def set(self, value):
            return None

        def time(self):
            return self

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

    def Counter(*args, **kwargs):
        return _NoopMetric()

    def Gauge(*args, **kwargs):
        return _NoopMetric()

    def Histogram(*args, **kwargs):
        return _NoopMetric()

    def generate_latest():
        return b"# prometheus_client is not installed\n"


# Настраиваем логи
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Energy Forecast API")

MODEL_LOADED = False
BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = BASE_DIR / "models" / "model.json"
TRAIN_SCRIPT = BASE_DIR / "src" / "models" / "train_optuna.py"
DEFAULT_MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

PREDICTION_REQUESTS = Counter(
    "energy_prediction_requests_total",
    "Total prediction requests.",
)
PREDICTION_ERRORS = Counter(
    "energy_prediction_errors_total",
    "Total failed prediction requests.",
)
PREDICTION_LATENCY = Histogram(
    "energy_prediction_latency_seconds",
    "Prediction request latency.",
)
LAST_PREDICTED_PRICE = Gauge(
    "energy_last_predicted_price",
    "Last predicted energy price.",
)
MODEL_LOADED_GAUGE = Gauge(
    "energy_model_loaded",
    "Model loaded status: 1 loaded, 0 missing or failed.",
)

# Загружаем модель
model = xgb.XGBRegressor()
if MODEL_PATH.exists():
    try:
        model.load_model(str(MODEL_PATH))
        MODEL_LOADED = True
        MODEL_LOADED_GAUGE.set(1)
        logger.info(f"Model loaded successfully from {MODEL_PATH}")
    except Exception as e:
        MODEL_LOADED_GAUGE.set(0)
        logger.error(f"Failed to load model: {e}")
else:
    MODEL_LOADED_GAUGE.set(0)
    logger.warning("Model file not found!")


# Схема данных (все те фичи, которые требовал XGBoost)
class PredictionInput(BaseModel):
    hour_sin: float
    hour_cos: float
    day_of_week: int
    is_holiday: int
    is_weekend: int
    load_forecast: float
    net_load_forecast: float
    solar_forecast: float
    wind_forecast: float
    renewable_total: float
    non_renewable_needed: float
    load_trend_24h: float
    price_fr_lag_24: float
    price_de_lag_24: float
    price_nl_lag_24: float
    spread_be_fr_lag_24: float
    spread_be_de_lag_24: float
    spread_be_nl_lag_24: float
    temperature_2m: float
    wind_speed_10m: float
    direct_radiation: float
    price_lag_24: float
    price_lag_48: float
    price_lag_168: float
    price_mean_24h: float
    price_std_24h: float


class RetrainRequest(BaseModel):
    dataset: str = "data/processed/energy_ready.csv"
    force: bool = False


class RetrainResponse(BaseModel):
    status: str
    job_id: str
    command: str
    tracking_uri: str
    dataset: str
    force: bool


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL_LOADED}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict")
def predict(input_data: PredictionInput):
    PREDICTION_REQUESTS.inc()
    try:
        with PREDICTION_LATENCY.time():
            # Преобразуем Pydantic модель в DataFrame
            X = pd.DataFrame([input_data.model_dump()])

            # Предсказание
            pred_log = model.predict(X)

            # Обратное преобразование (exp(x) - 1) и учет смещения OFFSET=50
            # Важно: убедись, что OFFSET вычитается именно в таком порядке
            final_price = np.expm1(pred_log) - 50

        predicted_price = float(final_price[0])
        LAST_PREDICTED_PRICE.set(predicted_price)
        return {"predicted_price": predicted_price}

    except Exception as e:
        PREDICTION_ERRORS.inc()
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=500, detail=f"Model prediction failed: {str(e)}"
        )


@app.post("/v1/retrain", response_model=RetrainResponse, status_code=202)
def retrain(request: RetrainRequest):
    if not TRAIN_SCRIPT.exists():
        raise HTTPException(status_code=404, detail="Training script not found")

    job_id = uuid.uuid4().hex
    command = [sys.executable, str(TRAIN_SCRIPT)]
    env = os.environ.copy()
    env["MLFLOW_TRACKING_URI"] = DEFAULT_MLFLOW_TRACKING_URI

    try:
        subprocess.Popen(
            command,
            cwd=str(BASE_DIR),
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception as e:
        logger.error(f"Retrain start error: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to start retrain: {str(e)}"
        )

    return RetrainResponse(
        status="started",
        job_id=job_id,
        command=" ".join(command),
        tracking_uri=DEFAULT_MLFLOW_TRACKING_URI,
        dataset=request.dataset,
        force=request.force,
    )
