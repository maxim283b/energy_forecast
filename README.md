Energy Forecast
MLOps project for electricity consumption prediction.

Инфраструктура проекта
Python: 3.14

MLflow: Настроен Docker Compose для автоматического трекинга метрик и моделей.

DVC: Внедрена система контроля версий данных (Data Version Control).

Быстрый старт
1. Настройка окружения

# Bash
git clone https://github.com/maxim283b/energy_forecast.git
cd energy_forecast

# Создание виртуального окружения
python -m venv venv
source venv/bin/activate  # Для Mac/Linux
# venv\Scripts\activate   # Для Windows

# Установка зависимостей
``` pip install -r requirements.txt

2. Запуск сервисов

Bash
# Запуск MLflow сервера (требуется Docker)
docker-compose up -d
Интерфейс MLflow доступен по адресу: http://localhost:5000

3. Проверка системы

Запустите файл test_infra.py через кнопку Run в VS Code. Сообщение Connection successful to MLflow! подтверждает готовность среды.

Структура папок
Plaintext
├── data
│   ├── processed      <- Финальные наборы данных для моделирования.
│   └── raw            <- Исходные неизменяемые данные (immuttable).
├── models             <- Обученные модели и их описания.
├── notebooks          <- Jupyter notebooks для анализа (EDA).
├── src                <- Исходный код проекта:
│   ├── data           <- Скрипты загрузки/генерации данных.
│   ├── features       <- Скрипты генерации признаков.
│   ├── models         <- Скрипты обучения и предсказания.
│   └── visualization  <- Скрипты визуализации результатов.
├── docker-compose.yml <- Конфигурация MLflow сервера.
└── README.md          <- Инструкция по использованию проекта.
