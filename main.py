# -*- coding: utf-8 -*-
import subprocess
import sys
import os
from pathlib import Path

# Определение корневой директории проекта
ROOT_DIR = Path(__file__).resolve().parent

def run_script(script_path, args=None):
    """
    Запускает скрипт Python как отдельный процесс.
    """
    full_path = ROOT_DIR / script_path
    
    # Проверка существования файла скрипта
    if not full_path.exists():
        print(f"Ошибка: Скрипт не найден по пути {full_path}")
        sys.exit(1)

    cmd = [sys.executable, str(full_path)]
    if args:
        cmd.extend(args)
    
    print(f"--- Запуск этапа: {script_path} ---")
    
    try:
        # Запуск процесса с передачей вывода в текущий терминал
        subprocess.run(cmd, check=True, text=True)
        print(f"--- Этап {script_path} успешно завершен --- \n")
    except subprocess.CalledProcessError as e:
        print(f"Ошибка при выполнении {script_path}: {e}")
        sys.exit(1)

def main():
    """
    Основной пайплайн проекта.
    Последовательность: Сбор -> Очистка -> Признаки -> Обучение
    """
    
    # 1. Загрузка данных (Data Ingestion)
    # Создает файл data/raw/big_energy_dataset_v2.csv
    run_script("src/data/data_loader.py")

    # 2. Очистка данных (Data Cleaning)
    # ПРОВЕРКА: Какое имя файла выдал загрузчик?
    raw_v1 = "data/raw/merged_energy_weather_2024.csv"
    raw_v2 = "data/raw/big_energy_dataset_v2.csv"
    
    # Выбираем тот файл, который реально существует на диске
    if (ROOT_DIR / raw_v2).exists():
        input_raw = raw_v2
    elif (ROOT_DIR / raw_v1).exists():
        input_raw = raw_v1
    else:
        print(f"Ошибка: Сырые данные не найдены в {raw_v1} или {raw_v2}")
        sys.exit(1)

    interim_data = "data/interim/energy_cleaned.csv"
    run_script("src/data/make_dataset.py", [input_raw, interim_data])

    # 3. Генерация признаков (Feature Engineering)
    run_script("src/features/build_features.py")

    # 4. Обучение модели (Optuna + Log Transformation)
    # Убедись, что файл называется train_optuna.py, как мы писали ранее
    run_script("src/models/train_optuna.py")

    # 5. Проверка инференса
    # Если ты еще не переименовал predict_model.py, оставь это имя
    if (ROOT_DIR / "src/models/predict_model.py").exists():
        run_script("src/models/predict_model.py")

    print("Пайплайн MLOps успешно выполнен полностью.")

if __name__ == "__main__":
    # Проверка и создание необходимых директорий
    for folder in ["data/raw", "data/interim", "data/processed", "data/models", "reports/figures"]:
        os.makedirs(ROOT_DIR / folder, exist_ok=True)
        
    main()