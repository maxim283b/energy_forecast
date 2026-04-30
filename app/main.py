from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import xgboost as xgb
import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Настраиваем логи
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Energy Forecast API")

MODEL_PATH = Path("data/models/model.json")

# Загружаем модель
model = xgb.XGBRegressor()
if MODEL_PATH.exists():
    try:
        model.load_model(str(MODEL_PATH))
        logger.info(f"Model loaded successfully from {MODEL_PATH}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
else:
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

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL_PATH.exists()}

@app.post("/predict")
def predict(input_data: PredictionInput):
    try:
        # Преобразуем Pydantic модель в DataFrame
        X = pd.DataFrame([input_data.model_dump()])
        
        # Предсказание
        pred_log = model.predict(X)
        
        # Обратное преобразование (exp(x) - 1) и учет смещения OFFSET=50
        # Важно: убедись, что OFFSET вычитается именно в таком порядке
        final_price = np.expm1(pred_log) - 50
        
        return {"predicted_price": float(final_price[0])}
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Model prediction failed: {str(e)}")