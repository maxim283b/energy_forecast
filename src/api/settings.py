import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = BASE_DIR / "models" / "model.json"
TRAIN_SCRIPT = BASE_DIR / "src" / "models" / "train_optuna.py"
RETRAIN_LOG_DIR = BASE_DIR / "reports" / "retrain"
DEFAULT_MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
DEFAULT_REGISTERED_MODEL_NAME = os.getenv("MLFLOW_REGISTERED_MODEL_NAME", "EnergyForecastXGBoost")
