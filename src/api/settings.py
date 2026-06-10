import os
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(BASE_DIR / "models" / "model.json"))).resolve()
TRAIN_SCRIPT = BASE_DIR / "src" / "models" / "train_optuna.py"
RETRAIN_LOG_DIR = BASE_DIR / "reports" / "retrain"
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
INTERIM_DATA_PATH = BASE_DIR / "data" / "interim" / "energy_cleaned.csv"
PROCESSED_DATA_PATH = BASE_DIR / "data" / "processed" / "energy_ready.csv"
PREDICTIONS_PATH = BASE_DIR / "data" / "predictions" / "latest_forecast.csv"
REPORTS_DIR = BASE_DIR / "reports" / "drift"
DEFAULT_MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
DEFAULT_REGISTERED_MODEL_NAME = os.getenv("MLFLOW_REGISTERED_MODEL_NAME", "EnergyForecastXGBoost")
DEFAULT_ENTSOE_COUNTRY_CODE = os.getenv("ENTSOE_COUNTRY_CODE", "BE")
DEFAULT_ENTSOE_LAT = float(os.getenv("ENTSOE_LAT", "50.85"))
DEFAULT_ENTSOE_LON = float(os.getenv("ENTSOE_LON", "4.35"))
DEFAULT_ENTSOE_START_YEAR = int(os.getenv("ENTSOE_START_YEAR", str(datetime.now().year - 1)))
DEFAULT_ENTSOE_END_YEAR = int(os.getenv("ENTSOE_END_YEAR", str(datetime.now().year)))
