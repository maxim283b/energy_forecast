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

# Установка зависимостей
pip install -r requirements.txt
```

### 2. Запуск сервисов 

``` bash
# Запуск MLflow сервера (требуется Docker)
docker-compose up -d
# Интерфейс MLflow доступен по адресу: http://localhost:5050
```

### 3. Запуск пайплайна

Для автоматического сбора данных и переобучения модели используйте одну команду:

``` bash
dvc repro
```
DVC проверит зависимости и запустит только измененные этапы (ingest -> train).

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

### 6. Предсказание (Inference)
После регистрации модели в MLflow, вы можете получить прогноз на следующий час:

``` bash
python src/inference/predict.py
```

### 7. Текущие метрики
Благодаря Feature Engineering (циклические признаки и лаги), достигнуты следующие показатели:

1. R2 Score: 0.637

2. MAE: 14.42 EUR/MWh

3. Model: RandomForest (depth=12, n_estimators=300)


### 8. Трекинг в MLflow

Для ручного логирования новых экспериментов используется стандартный блок:

``` python
import mlflow

mlflow.set_tracking_uri("http://localhost:5050")
mlflow.set_experiment("energy_prediction_optimized")

with mlflow.start_run():
    mlflow.log_param("n_estimators", 300)
    mlflow.log_metric("r2", 0.637)
```
