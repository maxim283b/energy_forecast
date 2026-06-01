# Используем легкую версию Python
FROM python:3.11-slim

# Устанавливаем рабочую директорию внутри контейнера
WORKDIR /app

# Устанавливаем системные зависимости для XGBoost и очистки
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Копируем файл зависимостей и устанавливаем их
COPY requirements.txt .
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel \
    && python -m pip install --no-cache-dir --timeout 120 --retries 10 -r requirements.txt

# Копируем исходный код и модель
# Важно: Docker должен видеть папку src и data
COPY src/ ./src/
COPY data/ ./data/
COPY models/ ./models/

# Открываем порт для FastAPI
EXPOSE 8000

# Команда для запуска сервера
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
