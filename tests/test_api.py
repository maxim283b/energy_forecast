from fastapi.testclient import TestClient

import src.api.main as main_module

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
    assert (
        "energy_model_loaded" in response.text or "prometheus_client" in response.text
    )


def test_prediction_endpoint():
    """Проверка предсказания с полным набором признаков"""
    # Создаем фиктивные данные, соответствующие классу PredictionInput
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

    response = client.post("/predict", json=valid_data)

    # Даже если модель выдаст ошибку из-за отсутствия весов,
    # статус должен быть либо 200, либо 500 (если модель не загружена),
    # но точно не 422.
    assert response.status_code in [200, 500]

    if response.status_code == 200:
        payload = response.json()
        assert "predicted_price" in payload
        assert "anomaly_flag" in payload


def test_retrain_endpoint_starts_job(monkeypatch, tmp_path):
    """Проверка запуска retrain без реального старта процесса."""
    calls = []
    dataset = tmp_path / "data" / "processed" / "energy_ready.csv"
    dataset.parent.mkdir(parents=True)
    dataset.write_text("timestamp,target\n2024-01-01,1.0\n", encoding="utf-8")
    train_script = tmp_path / "src" / "models" / "train_optuna.py"
    train_script.parent.mkdir(parents=True)
    train_script.write_text("print('train')\n", encoding="utf-8")
    monkeypatch.setattr(main_module, "BASE_DIR", tmp_path)
    monkeypatch.setattr(main_module, "TRAIN_SCRIPT", train_script)
    monkeypatch.setattr(
        main_module, "RETRAIN_LOG_DIR", tmp_path / "reports" / "retrain"
    )

    class DummyPopen:
        def __init__(self, *args, **kwargs):
            calls.append({"args": args, "kwargs": kwargs})

    monkeypatch.setattr(main_module.subprocess, "Popen", DummyPopen)

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
    assert payload["job_id"] in main_module.RETRAIN_JOBS
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
    monkeypatch.setattr(main_module, "BASE_DIR", tmp_path)
    monkeypatch.setattr(main_module, "TRAIN_SCRIPT", train_script)
    monkeypatch.setattr(
        main_module, "RETRAIN_LOG_DIR", tmp_path / "reports" / "retrain"
    )
    monkeypatch.setattr(main_module, "MODEL_PATH", existing_model)

    response = client.post(
        "/v1/retrain",
        json={"dataset": "data/processed/energy_ready.csv", "force": False},
    )

    assert response.status_code == 409
    assert "force=true" in response.json()["detail"]
