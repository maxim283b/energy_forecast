# Energy Forecast
MLOps-проект для прогноза цены электроэнергии на базе XGBoost, DVC, MLflow, FastAPI и Minikube.

## Что уже есть
- Python 3.11.
- Пайплайн данных: `ingest -> clean -> featurize -> train -> predict -> monitor`.
- MLflow для трекинга экспериментов, метрик, регистрации модели и drift-отчётов.
- DVC для версионирования данных и воспроизводимого запуска пайплайна.
- FastAPI-сервис с `/health`, `/metrics`, `/predict`, `POST /v1/retrain`,
  `GET /v1/retrain/{job_id}`, `POST /v1/model/reload` и admin upload endpoint
  для загрузки новых данных.
- Kubernetes CronJob для планового retrain внутри кластера.
- Streamlit UI для прогнозов, drift-отчётов, качества модели, MLflow experiments,
  drift-уведомлений, ручного retrain и admin upload новых датасетов.
- Prometheus и Grafana для runtime-метрик API.
- Helm chart для запуска API, Streamlit UI, MLflow и мониторинга в Kubernetes.

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

### 2. Восстановление DVC-артефактов
```bash
dvc pull models/model.json
```

Без `models/model.json` API запустится, но `/predict` вернет `503`.

### 3. Запуск всего проекта в Minikube
```bash
minikube start
eval $(minikube docker-env)
docker build -t energy-api:local .
kubectl create namespace energy-forecast
helm upgrade --install energy-api helm/energy-api \
  -f helm/energy-api/values-minikube.yaml \
  --namespace energy-forecast
```

Получение адресов сервисов:

- API: `minikube service energy-api -n energy-forecast --url`
- UI: `minikube service energy-api-ui -n energy-forecast --url`
- MLflow: `minikube service energy-api-mlflow -n energy-forecast --url`
- Prometheus: `minikube service energy-api-prometheus -n energy-forecast --url`
- Grafana: `minikube service energy-api-grafana -n energy-forecast --url`

Что входит в релиз:

- FastAPI API
- Streamlit UI
- MLflow
- Prometheus
- Grafana
- Kubernetes CronJob для retrain

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
curl -X POST "$(minikube service energy-api --url)/v1/retrain" \
  -H "Content-Type: application/json" \
  -d '{"dataset":"data/processed/energy_ready.csv","force":true}'
```

Ответ:
```json
{
  "status": "started",
  "job_id": "<id>",
  "command": "python .../src/models/train_optuna.py",
  "tracking_uri": "http://energy-api-mlflow:5000",
  "dataset": "data/processed/energy_ready.csv",
  "force": true
}
```

Статус фонового переобучения:
```bash
curl "$(minikube service energy-api --url)/v1/retrain/<job_id>"
```

После успешного retrain API автоматически перечитывает `models/model.json`.
Ручной reload модели:
```bash
curl -X POST "$(minikube service energy-api --url)/v1/model/reload"
```

В Kubernetes основным механизмом retrain должен быть не этот API, а `CronJob`.

### 5.1. Загрузка новых данных через admin interface
В UI добавлен `Admin` tab. Он отправляет CSV в API:

```bash
curl -X POST "$(minikube service energy-api --url)/v1/admin/dataset/upload" \
  -F "file=@data/raw/new_dataset.csv"
```

После загрузки API:

- сохраняет файл в `data/raw/`
- прогоняет `clean -> featurize`
- обновляет `data/interim/energy_cleaned.csv`
- обновляет `data/processed/energy_ready.csv`

Это подготавливает данные для следующего retrain.

### 7. Локальный инференс
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

UI показывает drift-уведомление на главной странице по состоянию
`reports/drift/retrain_trigger.json`.

Автоматический retrain по порогам:

```bash
AUTO_RETRAIN=true python src/monitoring/generate_reports.py
```

Плановый запуск:

```bash
0 2 * * * /path/to/energy_forecast/scripts/run_drift_monitor.sh >> /path/to/energy_forecast/reports/drift/cron.log 2>&1
```

### 8. Retrain через Kubernetes CronJob

В Helm chart добавлен `CronJob`, который:

- по расписанию запускает drift-monitoring
- если drift-пороги превышены, запускает `src/models/train_optuna.py`
- сохраняет модель в общий PVC
- вызывает `POST /v1/model/reload`, чтобы API перечитал новую модель

Ключевые values:

- `modelPersistence.enabled`
- `modelPersistence.size`
- `modelPersistence.mountPath`
- `retrainCron.enabled`
- `retrainCron.schedule`
- `retrainCron.dataset`
- `retrainCron.force`

Пример:

```bash
helm upgrade --install energy-api helm/energy-api \
  --namespace energy-forecast \
  --set retrainCron.enabled=true \
  --set retrainCron.schedule="0 2 * * *"
```

## MLflow
Tracking URI:

- внутри Kubernetes: `http://<release>-energy-api-mlflow:5000`

Модель и метрики логируются в MLflow вручную из `src/models/train_optuna.py`:

- params
- metrics
- model artifact
- registered model `EnergyForecastXGBoost`

В Streamlit UI есть вкладка `Experiments`, которая читает последние runs из
MLflow experiment `Energy_Forecast_Final`.

## DVC
Текущий remote:

- `storage`
- `gdrive://1cQPF0AXQ5FrwZTMjFsoFMbn0qnjGxSDR/dvcstore`

Для локальной работы и CI используются Google Drive OAuth credentials:

- `GDRIVE_CLIENT_ID`
- `GDRIVE_CLIENT_SECRET`
- `GDRIVE_REFRESH_TOKEN`

CI восстанавливает `models/model.json` через `dvc pull` перед сборкой Docker-образа.

## Эндпоинты
- `GET /health`
- `GET /metrics`
- `POST /predict`
- `POST /v1/model/reload`
- `POST /v1/retrain`
- `GET /v1/retrain/{job_id}`
- `POST /v1/admin/dataset/upload`

## Проверка качества
```bash
isort . --settings-path .isort.cfg --check-only
black . --config .black --check
flake8 --config .flake8 .
python3 -m pytest tests -q
python3 -m compileall src tests test_infra.py test_environment.py
```

## Deploy
Production-like GitOps flow:

- Pull Request в `main` запускает lint, tests, Docker build без push и Helm
  validation.
- Merge/push в `main` публикует Docker image в GHCR и обновляет Helm image tag.
- ArgoCD читает `main` и синхронизирует Helm chart в Kubernetes.

Прямой деплой Helm:
```bash
helm upgrade --install energy-api helm/energy-api \
  -f helm/energy-api/values-minikube.yaml \
  --namespace energy-forecast
```

GitOps через ArgoCD в Minikube:
```bash
eval $(minikube docker-env)
docker build -t energy-api:local .
kubectl create namespace argocd
kubectl apply -n argocd -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml
kubectl rollout status deployment/argocd-server -n argocd
kubectl apply -f helm/argocd-api-app-minikube.yaml
```

Открыть ArgoCD:
```bash
kubectl port-forward svc/argocd-server -n argocd 8081:443
```

UI ArgoCD будет доступен на:
```bash
https://localhost:8081
```

Начальный пароль admin:
```bash
kubectl -n argocd get secret argocd-initial-admin-secret \
  -o jsonpath="{.data.password}" | base64 --decode && echo
```

## Мониторинг
- Evidently генерирует data drift, target drift и regression quality отчёты.
- PSI и деградация MAE/R2 формируют `reports/drift/retrain_trigger.json`.
- Prometheus собирает `/metrics` FastAPI: request count, latency, errors,
  model loaded status, last predicted price и anomaly count.
- Grafana автоматически подхватывает dashboard `Energy Forecast Monitoring`.
