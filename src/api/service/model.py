import logging
from datetime import datetime, timezone
from time import perf_counter

import numpy as np
import pandas as pd
import xgboost as xgb

from src.api import settings
from src.api.metrics import (
    LAST_PREDICTED_PRICE,
    MODEL_FILE_MTIME_TIMESTAMP,
    MODEL_FILE_SIZE_BYTES,
    MODEL_LAST_RELOAD_TIMESTAMP,
    MODEL_LOADED_GAUGE,
    MODEL_RELOAD_DURATION,
    MODEL_RELOADS,
    PREDICTION_ANOMALIES,
    PREDICTION_ERRORS,
    PREDICTION_LATENCY,
    PREDICTION_REQUESTS,
)
from src.api.schemas import PredictionInput, PredictionResponse

logger = logging.getLogger(__name__)

MODEL_LOADED = False
model = xgb.XGBRegressor()


class ModelNotLoadedError(RuntimeError):
    pass


class ModelPredictionError(RuntimeError):
    pass


def is_model_loaded() -> bool:
    return MODEL_LOADED


def reload_model() -> bool:
    global MODEL_LOADED, model

    model = xgb.XGBRegressor()
    started_at = perf_counter()
    if not settings.MODEL_PATH.exists():
        MODEL_LOADED = False
        MODEL_LOADED_GAUGE.set(0)
        MODEL_RELOADS.labels(result="missing").inc()
        MODEL_RELOAD_DURATION.observe(perf_counter() - started_at)
        logger.warning("Model file not found: %s", settings.MODEL_PATH)
        return False

    try:
        model.load_model(str(settings.MODEL_PATH))
        MODEL_LOADED = True
        MODEL_LOADED_GAUGE.set(1)
        MODEL_RELOADS.labels(result="success").inc()
        MODEL_RELOAD_DURATION.observe(perf_counter() - started_at)
        MODEL_LAST_RELOAD_TIMESTAMP.set(datetime.now(timezone.utc).timestamp())
        MODEL_FILE_SIZE_BYTES.set(settings.MODEL_PATH.stat().st_size)
        MODEL_FILE_MTIME_TIMESTAMP.set(settings.MODEL_PATH.stat().st_mtime)
        logger.info("Model loaded successfully from %s", settings.MODEL_PATH)
        return True
    except Exception:
        MODEL_LOADED = False
        MODEL_LOADED_GAUGE.set(0)
        MODEL_RELOADS.labels(result="error").inc()
        MODEL_RELOAD_DURATION.observe(perf_counter() - started_at)
        logger.exception("Failed to load model")
        return False


def predict_price(input_data: PredictionInput) -> PredictionResponse:
    PREDICTION_REQUESTS.inc()
    if not MODEL_LOADED:
        PREDICTION_ERRORS.inc()
        raise ModelNotLoadedError("Model is not loaded")

    try:
        with PREDICTION_LATENCY.time():
            features = pd.DataFrame([input_data.model_dump()])
            pred_log = model.predict(features)
            final_price = np.expm1(pred_log) - 50

        predicted_price = float(final_price[0])
        anomaly_flag = predicted_price < 0 or predicted_price > 200
        LAST_PREDICTED_PRICE.set(predicted_price)
        if anomaly_flag:
            PREDICTION_ANOMALIES.inc()

        return PredictionResponse(predicted_price=predicted_price, anomaly_flag=anomaly_flag)
    except Exception as exc:
        PREDICTION_ERRORS.inc()
        logger.exception("Prediction error")
        raise ModelPredictionError(str(exc)) from exc


reload_model()
