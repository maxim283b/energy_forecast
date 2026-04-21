import subprocess
import sys
import os
from pathlib import Path

# Определение корневой директории проекта
ROOT_DIR = Path(__file__).resolve().parent

def run_script(script_path, args=None):
    """
    Запускает скрипт Python как отдельный процесс.
    
    Args:
        script_path (str): Путь к скрипту относительно корня проекта.
        args (list): Список дополнительных аргументов командной строки.
    """
    cmd = [sys.executable, str(ROOT_DIR / script_path)]
    if args:
        cmd.extend(args)
    
    print(f"--- Запуск этапа: {script_path} ---")
    
    try:
        # Запуск процесса с передачей вывода в текущий терминал
        result = subprocess.run(cmd, check=True, text=True)
        print(f"--- Этап {script_path} успешно завершен --- \n")
    except subprocess.CalledProcessError as e:
        print(f"Ошибка при выполнении {script_path}: {e}")
        sys.exit(1)

def main():
    """
    Основной пайплайн проекта.
    Последовательность: Сбор -> Очистка -> Признаки -> Обучение -> Тест
    """
    
    # 1. Загрузка данных (Data Ingestion)
    # Сохраняет сырые данные в data/raw/merged_energy_weather_2024.csv
    run_script("src/data/data_loader.py")

    # 2. Очистка данных (Data Cleaning)
    # Использует click аргументы: входной файл и выходной файл (interim)
    raw_data = "data/raw/merged_energy_weather_2024.csv"
    interim_data = "data/interim/energy_cleaned.csv"
    run_script("src/data/make_dataset.py", [raw_data, interim_data])

    # 3. Генерация признаков (Feature Engineering)
    # Превращает очищенные данные в готовый датасет для модели в data/processed
    run_script("src/features/build_features.py")

    # 4. Обучение модели (Путь изменен на src/models/)
    run_script("src/models/train_final.py")

    # 5. Проверка инференса (Путь изменен на src/models/)
    run_script("src/models/predict_model.py")

    #6. Визуализация результатов
    run_script("src/visualization/visualize.py")

    print("Пайплайн MLOps успешно выполнен полностью.")

if __name__ == "__main__":
    # Проверка наличия директорий данных перед запуском
    for folder in ["data/raw", "data/interim", "data/processed"]:
        os.makedirs(ROOT_DIR / folder, exist_ok=True)
        
    main()