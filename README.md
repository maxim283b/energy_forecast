# Energy Forecast
MLOps project for electricity price prediction using Random Forest and automated pipelines.

## Инфраструктура проекта
* **Python:** 3.14 (зависимости зафиксированы в `requirements.txt`).
* **MLflow:** Развернут через Docker Compose для трекинга экспериментов и управления реестром моделей (Model Registry).
* **DVC:** Внедрена система контроля версий данных и автоматизации пайплайнов (Data Version Control).

## Быстрый старт

### 1. Настройка окружения
```bash
# Клонирование репозитория
git clone [https://github.com/maxim283b/energy_forecast.git](https://github.com/maxim283b/energy_forecast.git)
cd energy_forecast

# Создание виртуального окружения
python -m venv venv
source venv/bin/activate  # Для Mac/Linux
# venv\Scripts\activate   # Для Windows

# Установка зависимостей (рекомендуется для локальной работы, Python 3.12)
pip install --upgrade pip setuptools wheel
pip install -r requirements-local.txt

# Полный requirements.txt от коллеги конфликтует на Python 3.12
# (mlflow 2.7.1 требует pyarrow<14, а wheels для 3.12 — только pyarrow 14+).
```

### 2. Запуск сервисов 

``` bash
# Запуск MLflow сервера (требуется Docker)
docker-compose up -d
# Интерфейс MLflow: http://localhost:5050 (на macOS порт 5000 часто занят AirPlay)
```

### 3. Запуск пайплайна

Для автоматического сбора данных и переобучения модели используйте одну команду:

``` bash
dvc repro
```
DVC проверит зависимости и запустит только измененные этапы (ingest -> train).

Теперь пайплайн включает этапы:
- `ingest` -> `clean` -> `features` -> `train` -> `predict` -> `monitor`
- На этапе `monitor` генерируются отчеты Evidently в `reports/drift/` и дублируются в MLflow artifacts.

### 4. Структура проекта

``` text
├── data/
│   ├── processed/      <- Обработанные данные (готовые для обучения).
│   └── raw/            <- Исходные данные из API (под контролем DVC).
├── docs/               <- Документация проекта (Sphinx/RST).
├── models/             <- Место хранения локальных весов моделей.
├── notebooks/          <- Jupyter Notebooks для EDA и черновиков.
├── reports/            <- Отчеты и сгенерированные графики (figures).
├── src/                <- Исходный код:
│   ├── data_ingestion/ <- Парсеры (entsoe_parser.py, open_meteo.py).
│   ├── training/       <- Скрипты обучения (train_optimized.py и др.).
│   ├── inference/      <- Скрипт для предсказаний (predict.py).
│   ├── features/       <- Скрипты генерации признаков.
│   └── visualization/  <- Код для построения графиков.
├── dvc.yaml            <- Конфигурация пайплайна.
├── dvc.lock            <- Фиксация состояний данных.
├── docker-compose.yml  <- Запуск MLflow сервера.
└── README.md           <- Эта инструкция.
```

## Использование и результаты

### 1. Предсказание (Inference)
После регистрации модели в MLflow, вы можете получить прогноз на следующий час:

``` bash
python src/inference/predict.py
```

### 2. Локальный мониторинг и UI

``` bash
# 1) Сгенерировать отчеты Evidently локально
python src/monitoring/generate_reports.py

# 2) Запустить локальный Web UI
streamlit run src/ui/app.py
```

Dashboard показывает:
- историю последних предиктов (`data/predictions/latest_forecast.csv`);
- отчеты Data Drift / Target Drift;
- отчет качества Regression Quality;
- статус триггера переобучения и кнопку `Start Retraining` (POST `http://127.0.0.1:8001/v1/retrain`).

Пороги автотриггера (локально в `reports/drift/retrain_trigger.json`):
- PSI > 0.2 по ключевому признаку или target;
- MAE вырос > 10% от baseline;
- R2 упал > 0.05 от baseline.

Автозапуск переобучения при срабатывании порогов:
```bash
AUTO_RETRAIN=true python src/monitoring/generate_reports.py
```

Ежедневный мониторинг (02:00):
```bash
0 2 * * * /path/to/energy_forecast/scripts/run_drift_monitor.sh >> /path/to/energy_forecast/reports/drift/cron.log 2>&1
```

DVC remote (Google Drive):
```bash
export GDRIVE_CLIENT_ID=...
export GDRIVE_CLIENT_SECRET=...
export GDRIVE_REFRESH_TOKEN=...
dvc remote modify storage --local gdrive_client_id "$GDRIVE_CLIENT_ID"
dvc remote modify storage --local gdrive_client_secret "$GDRIVE_CLIENT_SECRET"
dvc remote modify storage --local gdrive_refresh_token "$GDRIVE_REFRESH_TOKEN"
```

### 1. Текущие метрики
Благодаря Feature Engineering (циклические признаки и лаги), достигнуты следующие показатели:

1. R2 Score: 0.637

2. MAE: 14.42 EUR/MWh

3. Model: RandomForest (depth=12, n_estimators=300)


### 3. Трекинг в MLflow

Для ручного логирования новых экспериментов используется стандартный блок:

``` python
import mlflow

mlflow.set_tracking_uri("http://localhost:5050")
mlflow.set_experiment("energy_prediction_optimized")

with mlflow.start_run():
    mlflow.log_param("n_estimators", 300)
    mlflow.log_metric("r2", 0.637)
```
