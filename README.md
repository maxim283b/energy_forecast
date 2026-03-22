# Energy Forecast

MLOps project for electricity consumption prediction.

## Инфраструктура проекта

* **Python: 3.14** (зависимости зафиксированы в `requirements.txt`).
* **MLflow:** Настроен Docker Compose для автоматического трекинга метрик и моделей.
* **DVC:** Внедрена система контроля версий данных (Data Version Control).
* **IDE:** Настроены локальные конфиги для VS Code (`.vscode`) для корректного запуска через кнопку Run.

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

### 2. Запуск сервисов

```bash
# Запуск MLflow сервера (требуется Docker)
docker-compose up -d
Интерфейс MLflow доступен по адресу: http://localhost:5000

### 3. Проверка системы

Для проверки корректности настройки запустите файл test_infra.py через кнопку Run в VS Code или через терминал:

```bash
python test_infra.py
Сообщение Connection successful to MLflow! подтверждает готовность среды к работе.

Структура проекта
Plaintext
├── data
│   ├── processed      <- Финальные наборы данных для моделирования.
│   └── raw            <- Исходные неизменяемые данные (immutable).
├── models             <- Обученные модели и их описания.
├── notebooks          <- Jupyter notebooks для анализа (EDA).
├── src                <- Исходный код проекта:
│   ├── data           <- Скрипты загрузки/генерации данных.
│   ├── features       <- Скрипты генерации признаков.
│   ├── models         <- Скрипты обучения и предсказания.
│   └── visualization  <- Скрипты визуализации результатов.
├── docker-compose.yml <- Конфигурация MLflow сервера.
└── README.md          <- Инструкция по использованию проекта.
Использование MLflow

## Для регистрации параметров и метрик в коде используйте следующий блок:

```python
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("energy_prediction")

with mlflow.start_run():
    mlflow.log_param("model_type", "random_forest")
    mlflow.log_metric("rmse", 0.12)
