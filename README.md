# Energy Forecast
MLOps-проект для прогноза цены электроэнергии на базе XGBoost с DVC, MLflow, FastAPI и Docker.

## Что уже есть
- Python 3.11.
- Пайплайн данных: `ingest -> clean -> featurize -> train -> predict -> monitor`.
- MLflow для трекинга экспериментов, метрик, модели и drift-отчётов.
- DVC для версионирования данных и воспроизводимого запуска пайплайна.
- FastAPI-сервис с `/health`, `/metrics`, `/predict` и `POST /v1/retrain`.
- Streamlit UI для прогнозов, drift-отчётов, качества модели и ручного retrain.
- Prometheus и Grafana для runtime-метрик API.
- Docker Compose для локального запуска MLflow, API, UI и мониторинга.
- Helm chart для деплоя API и MLflow через ArgoCD.

## Последние результаты модели
Последний сохранённый run в MLflow:

- `MAE`: `16.6152`
- `R2`: `0.7552`
- `RMSE`: `23.6070`

Последняя модель сохранена в:

- `models/model.json`

Эксперимент MLflow:

- `Energy_Forecast_Final`

## Структура проекта
```text
├── data/
│   ├── raw/             <- исходные данные
│   ├── interim/         <- очищенные данные
│   ├── processed/       <- признаки для обучения
│   └── predictions/     <- локальные результаты инференса
├── docs/                <- документация
├── helm/                <- Helm chart и ArgoCD manifest
├── models/              <- сериализованные модели
├── reports/figures/     <- графики обучения и визуализации
├── reports/drift/       <- Evidently drift-отчёты
├── monitoring/          <- Prometheus и Grafana provisioning
├── src/
│   ├── api/             <- FastAPI приложение
│   ├── data/            <- загрузка и очистка данных
│   ├── features/        <- генерация признаков
│   ├── monitoring/      <- drift reports и retrain trigger
│   ├── models/          <- обучение и локальный инференс
│   ├── ui/              <- Streamlit dashboard
│   └── visualization/   <- графики и отчёты
├── tests/               <- API-тесты
├── dvc.yaml             <- DVC pipeline
├── dvc.lock             <- зафиксированное состояние пайплайна
├── docker-compose.yml   <- локальный запуск MLflow и API
└── README.md
```

## Быстрый старт

### 1. Установка зависимостей
```bash
git clone https://github.com/maxim283b/energy_forecast.git
cd energy_forecast

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

Если окружение уже использовалось раньше и в нем был другой `mlflow`, лучше сразу выровнять пакеты:

```bash
pip uninstall -y mlflow mlflow-skinny protobuf
pip install -r requirements.txt
```

### 2. Запуск локальной инфраструктуры
```bash
docker compose up -d mlflow_server energy_api streamlit_ui prometheus grafana
```

Локальные адреса:

- API: `http://localhost:8080`
- API metrics: `http://localhost:8080/metrics`
- MLflow: `http://localhost:5000`
- Streamlit UI: `http://localhost:8501`
- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000` (`admin` / `admin`)

### 3. Запуск API без Docker
```bash
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8001
```

Проверка health:
```bash
curl http://127.0.0.1:8001/health
```

Проверка Prometheus metrics:
```bash
curl http://127.0.0.1:8001/metrics
```

### 4. Запуск всего пайплайна
```bash
dvc repro
```

Команда последовательно запускает:

- `ingest`
- `clean`
- `featurize`
- `train`
- `predict`
- `monitor`

### 5. Переобучение через API
```bash
curl -X POST http://127.0.0.1:8001/v1/retrain \
  -H "Content-Type: application/json" \
  -d '{"dataset":"data/processed/energy_ready.csv","force":true}'
```

Ответ:
```json
{
  "status": "started",
  "job_id": "<id>",
  "command": "python .../src/models/train_optuna.py",
  "tracking_uri": "http://localhost:5000",
  "dataset": "data/processed/energy_ready.csv",
  "force": true
}
```

### 6. Локальный инференс
```bash
python src/models/predict_model.py
```

Скрипт берёт данные из `data/processed/energy_ready.csv`, делает прогноз и сохраняет результат в:

- `data/predictions/latest_forecast.csv`

### 7. Drift monitoring и UI
```bash
python src/monitoring/generate_reports.py
streamlit run src/ui/app.py
```

Drift-отчёты сохраняются в:

- `reports/drift/data_drift.html`
- `reports/drift/target_drift.html`
- `reports/drift/regression_quality.html`
- `reports/drift/retrain_trigger.json`

Автоматический retrain по порогам:

```bash
AUTO_RETRAIN=true python src/monitoring/generate_reports.py
```

Плановый запуск:

```bash
0 2 * * * /path/to/energy_forecast/scripts/run_drift_monitor.sh >> /path/to/energy_forecast/reports/drift/cron.log 2>&1
```

## MLflow
Tracking URI:

- локально: `http://localhost:5000`
- внутри Docker Compose: `http://mlflow_server:5000`
- внутри Kubernetes: `http://<release>-energy-api-mlflow:5000`

Модель и метрики логируются в MLflow вручную из `src/models/train_optuna.py`:

- params
- metrics
- model artifact

## DVC
Текущий remote:

- `storage`
- `gdrive://1cQPF0AXQ5FrwZTMjFsoFMbn0qnjGxSDR/dvcstore`

Для локальной работы и CI используются Google Drive OAuth credentials:

- `GDRIVE_CLIENT_ID`
- `GDRIVE_CLIENT_SECRET`
- `GDRIVE_REFRESH_TOKEN`

## Эндпоинты
- `GET /health`
- `GET /metrics`
- `POST /predict`
- `POST /v1/retrain`

## Проверка качества
```bash
python3 -m pytest tests -q
python3 -m compileall src tests test_infra.py test_environment.py
```

## Deploy
```bash
helm upgrade --install energy-api helm/energy-api --namespace default
kubectl apply -f helm/argocd-api-app.yaml
```

Для Minikube:
```bash
minikube service energy-api --url
minikube service energy-api-mlflow --url
```

## Мониторинг
- Evidently генерирует data drift, target drift и regression quality отчёты.
- PSI и деградация MAE/R2 формируют `reports/drift/retrain_trigger.json`.
- Prometheus собирает `/metrics` FastAPI.
- Grafana автоматически подхватывает dashboard `Energy Forecast Monitoring`.
