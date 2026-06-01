import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "Energy_Forecast_Final")

RETRAIN_API_URL = os.getenv("RETRAIN_API_URL", "http://127.0.0.1:8001/v1/retrain")
RETRAIN_DATASET = os.getenv("RETRAIN_DATASET", "data/processed/energy_ready.csv")

DATA_PATH = BASE_DIR / "data/processed/energy_ready.csv"
MODEL_PATH = BASE_DIR / "models/model.json"
PREDICTIONS_PATH = BASE_DIR / "data/predictions/latest_forecast.csv"
REPORTS_DIR = BASE_DIR / "reports/drift"
BASELINE_METRICS_PATH = REPORTS_DIR / "baseline_metrics.json"
TRIGGER_STATUS_PATH = REPORTS_DIR / "retrain_trigger.json"

PSI_THRESHOLD = float(os.getenv("PSI_THRESHOLD", "0.2"))
MAE_INCREASE_RATIO = float(os.getenv("MAE_INCREASE_RATIO", "0.10"))
R2_DROP_THRESHOLD = float(os.getenv("R2_DROP_THRESHOLD", "0.05"))

KEY_FEATURES = [
    "hour_sin",
    "hour_cos",
    "load_forecast",
    "price_lag_24",
    "price_lag_168",
    "temperature_2m",
    "wind_speed_10m",
    "target",
]
