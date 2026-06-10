import json
import os
import subprocess
import sys
from pathlib import Path

import requests

from src.monitoring import generate_reports
from src.monitoring.config import RETRAIN_DATASET, TRIGGER_STATUS_PATH

BASE_DIR = Path(__file__).resolve().parents[2]
TRAIN_SCRIPT = BASE_DIR / "src" / "models" / "train_optuna.py"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
RELOAD_MODEL_URL = os.getenv("RELOAD_MODEL_URL", "").strip()
FORCE_RETRAIN = os.getenv("FORCE_RETRAIN", "false").lower() == "true"


def load_trigger_status() -> dict:
    if not TRIGGER_STATUS_PATH.exists():
        raise FileNotFoundError(f"Retrain trigger file not found: {TRIGGER_STATUS_PATH}")
    return json.loads(TRIGGER_STATUS_PATH.read_text(encoding="utf-8"))


def maybe_reload_model() -> None:
    if not RELOAD_MODEL_URL:
        return
    response = requests.post(RELOAD_MODEL_URL, timeout=30)
    response.raise_for_status()


def main() -> int:
    os.environ["AUTO_RETRAIN"] = "false"
    generate_reports.main()
    trigger_status = load_trigger_status()

    should_retrain = FORCE_RETRAIN or trigger_status.get("should_retrain", False)
    if not should_retrain:
        print("Retrain skipped: trigger conditions not met.")
        return 0

    if not TRAIN_SCRIPT.exists():
        raise FileNotFoundError(f"Training script not found: {TRAIN_SCRIPT}")

    command = [sys.executable, str(TRAIN_SCRIPT), "--dataset", RETRAIN_DATASET, "--force"]
    env = os.environ.copy()
    env["MLFLOW_TRACKING_URI"] = MLFLOW_TRACKING_URI
    env["RETRAIN_DATASET"] = RETRAIN_DATASET

    print(f"Starting retrain: {' '.join(command)}")
    subprocess.run(command, cwd=BASE_DIR, env=env, check=True)
    maybe_reload_model()
    print("Retrain finished successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
