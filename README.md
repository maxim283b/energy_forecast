# Energy Forecast
MLOps project for electricity consumption prediction.

## Инфраструктура проекта
Python: 3.14 (зависимости зафиксированы в requirements.txt).

MLflow: Настроен Docker Compose для автоматического трекинга метрик и моделей.

DVC: Внедрена система контроля версий данных (Data Version Control).



## Быстрый старт
## 1. Настройка окружения

``` python
# Клонирование репозитория
git clone https://github.com/maxim283b/energy_forecast.git
cd energy_forecast
```

``` bash
# Создание виртуального окружения
python -m venv venv
source venv/bin/activate  # Для Mac/Linux
venv\Scripts\activate   # Для Windows
```

``` bash
#Установка зависимостей
pip install -r requirements.txt
```
## 2. Запуск сервисов

``` bash
# Запуск MLflow сервера (требуется Docker)
docker-compose up -d
# Интерфейс MLflow доступен по адресу: http://localhost:5000
```

## 3. Проверка системы

Запустите файл test_infra.py через IDE или через терминал:

``` bash
python test_infra.py
# Сообщение Connection successful to MLflow! подтверждает готовность среды к работе.
```
Структура проекта

``` text
├── data               <- Папки с данными (raw, processed).
├── models             <- Обученные модели и их веса.
├── src                <- Исходный код:
│   ├── data           <- Загрузка и очистка данных.
│   ├── features       <- Генерация признаков для модели.
│   └── models         <- Обучение и предсказание.
├── docker-compose.yml <- Конфигурация для запуска MLflow.
├── Makefile           <- Автоматизация команд (run, train, data).
├── requirements.txt   <- Список зависимостей проекта.
├── test_infra.py      <- Скрипт проверки готовности инфраструктуры.
└── README.md          <- Эта инструкция.
```

## Использование MLflow
Для регистрации параметров и метрик в коде используйте следующий блок:

``` python
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("energy_prediction")

with mlflow.start_run():
    mlflow.log_param("model_type", "random_forest")
    mlflow.log_metric("rmse", 0.12)
```
