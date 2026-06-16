import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

import src.api.main as main_module
from src.api import settings
from src.api.service import admin as admin_service
from src.api.service import model as model_service
from src.api.service import retrain as retrain_service

app = main_module.app

client = TestClient(app)


def test_health():
    """Проверка эндпоинта /health (в твоем коде он /health, а не /)"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "energy_model_loaded" in response.text or "prometheus_client" in response.text


def test_prediction_endpoint():
    """Проверка предсказания с полным набором признаков"""
    valid_data = {
        "hour_sin": 0.5,
        "hour_cos": 0.8,
        "day_of_week": 1,
        "is_holiday": 0,
        "is_weekend": 0,
        "load_forecast": 1000.0,
        "net_load_forecast": 800.0,
        "solar_forecast": 100.0,
        "wind_forecast": 50.0,
        "renewable_total": 150.0,
        "non_renewable_needed": 650.0,
        "load_trend_24h": 0.05,
        "price_fr_lag_24": 50.0,
        "price_de_lag_24": 45.0,
        "price_nl_lag_24": 48.0,
        "spread_be_fr_lag_24": 2.0,
        "spread_be_de_lag_24": 5.0,
        "spread_be_nl_lag_24": 2.0,
        "temperature_2m": 15.0,
        "wind_speed_10m": 5.0,
        "direct_radiation": 200.0,
        "price_lag_24": 50.0,
        "price_lag_48": 52.0,
        "price_lag_168": 48.0,
        "price_mean_24h": 50.0,
        "price_std_24h": 5.0,
    }

    class DummyModel:
        def predict(self, features):
            return np.array([np.log1p(150.0)])

    previous_model = model_service.model
    previous_loaded = model_service.MODEL_LOADED
    model_service.model = DummyModel()
    model_service.MODEL_LOADED = True

    response = client.post("/predict", json=valid_data)

    model_service.model = previous_model
    model_service.MODEL_LOADED = previous_loaded

    assert response.status_code == 200
    payload = response.json()
    assert payload["predicted_price"] == 100.0
    assert payload["anomaly_flag"] is False


def test_prediction_endpoint_requires_loaded_model():
    previous_loaded = model_service.MODEL_LOADED
    model_service.MODEL_LOADED = False

    response = client.post(
        "/predict",
        json={
            "hour_sin": 0.5,
            "hour_cos": 0.8,
            "day_of_week": 1,
            "is_holiday": 0,
            "is_weekend": 0,
            "load_forecast": 1000.0,
            "net_load_forecast": 800.0,
            "solar_forecast": 100.0,
            "wind_forecast": 50.0,
            "renewable_total": 150.0,
            "non_renewable_needed": 650.0,
            "load_trend_24h": 0.05,
            "price_fr_lag_24": 50.0,
            "price_de_lag_24": 45.0,
            "price_nl_lag_24": 48.0,
            "spread_be_fr_lag_24": 2.0,
            "spread_be_de_lag_24": 5.0,
            "spread_be_nl_lag_24": 2.0,
            "temperature_2m": 15.0,
            "wind_speed_10m": 5.0,
            "direct_radiation": 200.0,
            "price_lag_24": 50.0,
            "price_lag_48": 52.0,
            "price_lag_168": 48.0,
            "price_mean_24h": 50.0,
            "price_std_24h": 5.0,
        },
    )

    model_service.MODEL_LOADED = previous_loaded

    assert response.status_code == 503
    assert response.json()["detail"] == "Model is not loaded"


def test_retrain_endpoint_starts_job(monkeypatch, tmp_path):
    """Проверка запуска retrain без реального старта процесса."""
    calls = []
    dataset = tmp_path / "data" / "processed" / "energy_ready.csv"
    dataset.parent.mkdir(parents=True)
    dataset.write_text("timestamp,target\n2024-01-01,1.0\n", encoding="utf-8")
    train_script = tmp_path / "src" / "models" / "train_optuna.py"
    train_script.parent.mkdir(parents=True)
    train_script.write_text("print('train')\n", encoding="utf-8")
    monkeypatch.setattr(settings, "BASE_DIR", tmp_path)
    monkeypatch.setattr(settings, "TRAIN_SCRIPT", train_script)
    monkeypatch.setattr(settings, "RETRAIN_LOG_DIR", tmp_path / "reports" / "retrain")
    monkeypatch.setattr(settings, "MODEL_PATH", tmp_path / "models" / "model.json")
    retrain_service.RETRAIN_JOBS.clear()

    class DummyPopen:
        def __init__(self, *args, **kwargs):
            calls.append({"args": args, "kwargs": kwargs})

    monkeypatch.setattr(retrain_service.subprocess, "Popen", DummyPopen)

    response = client.post(
        "/v1/retrain",
        json={"dataset": "data/processed/energy_ready.csv", "force": True},
    )

    assert response.status_code == 202
    payload = response.json()
    assert payload["status"] == "started"
    assert payload["dataset"] == "data/processed/energy_ready.csv"
    assert payload["force"] is True
    assert payload["tracking_uri"] == "http://localhost:5000"
    assert payload["job_id"] in retrain_service.RETRAIN_JOBS
    assert calls
    command = calls[0]["args"][0]
    assert "--dataset" in command
    assert "--force" in command
    assert command[-1] == "--force"

    status_response = client.get(f"/v1/retrain/{payload['job_id']}")
    assert status_response.status_code == 200
    assert status_response.json()["status"] == "running"


def test_retrain_endpoint_requires_force_for_existing_model(monkeypatch, tmp_path):
    dataset = tmp_path / "data" / "processed" / "energy_ready.csv"
    dataset.parent.mkdir(parents=True)
    dataset.write_text("timestamp,target\n2024-01-01,1.0\n", encoding="utf-8")
    train_script = tmp_path / "src" / "models" / "train_optuna.py"
    train_script.parent.mkdir(parents=True)
    train_script.write_text("print('train')\n", encoding="utf-8")
    existing_model = tmp_path / "model.json"
    existing_model.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(settings, "BASE_DIR", tmp_path)
    monkeypatch.setattr(settings, "TRAIN_SCRIPT", train_script)
    monkeypatch.setattr(settings, "RETRAIN_LOG_DIR", tmp_path / "reports" / "retrain")
    monkeypatch.setattr(settings, "MODEL_PATH", existing_model)

    response = client.post(
        "/v1/retrain",
        json={"dataset": "data/processed/energy_ready.csv", "force": False},
    )

    assert response.status_code == 409
    assert "force=true" in response.json()["detail"]


def test_admin_dataset_upload_processes_pipeline(monkeypatch, tmp_path):
    raw_dir = tmp_path / "data" / "raw"
    raw_history_path = raw_dir / "big_energy_dataset_v2.csv"
    interim_path = tmp_path / "data" / "interim" / "energy_cleaned.csv"
    processed_path = tmp_path / "data" / "processed" / "energy_ready.csv"
    monkeypatch.setattr(settings, "BASE_DIR", tmp_path)
    monkeypatch.setattr(settings, "RAW_DATA_DIR", raw_dir)
    monkeypatch.setattr(settings, "RAW_HISTORY_PATH", raw_history_path)
    monkeypatch.setattr(settings, "INTERIM_DATA_PATH", interim_path)
    monkeypatch.setattr(settings, "PROCESSED_DATA_PATH", processed_path)

    class ImmediateThread:
        def __init__(self, target, args=(), kwargs=None, daemon=None):
            self.target = target
            self.args = args
            self.kwargs = kwargs or {}

        def start(self):
            self.target(*self.args, **self.kwargs)

    monkeypatch.setattr(admin_service.threading, "Thread", ImmediateThread)
    monkeypatch.setattr(admin_service, "_refresh_artifacts", lambda: (True, True))
    admin_service.ADMIN_JOBS.clear()

    rows = 220
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=rows, freq="h", tz="UTC").astype(str),
            "price": np.linspace(10, 100, rows),
            "load_forecast": np.linspace(1000, 1200, rows),
            "solar_forecast": np.linspace(50, 90, rows),
            "wind_forecast": np.linspace(40, 70, rows),
            "temperature_2m": np.linspace(5, 20, rows),
            "wind_speed_10m": np.linspace(2, 8, rows),
            "direct_radiation": np.linspace(0, 300, rows),
            "price_fr": np.linspace(12, 102, rows),
            "price_de": np.linspace(11, 101, rows),
            "price_nl": np.linspace(13, 103, rows),
        }
    )

    response = client.post(
        "/v1/admin/dataset/upload",
        files={"file": ("admin_upload.csv", df.to_csv(index=False), "text/csv")},
    )

    assert response.status_code == 201
    payload = response.json()
    assert payload["status"] == "started"
    assert payload["job_type"] == "dataset_upload"
    job_id = payload["job_id"]

    status_response = client.get(f"/v1/admin/jobs/{job_id}")
    assert status_response.status_code == 200
    job_payload = status_response.json()
    assert job_payload["status"] == "succeeded"
    assert job_payload["result"]["status"] == "uploaded"
    assert job_payload["result"]["ready_for_retrain"] is True
    assert job_payload["result"]["processed_rows"] > 0
    assert raw_dir.exists()
    assert raw_history_path.exists()
    assert interim_path.exists()
    assert processed_path.exists()
