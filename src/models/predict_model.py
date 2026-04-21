import pandas as pd
import numpy as np
import xgboost as xgb
from pathlib import Path

# Определение путей относительно корня проекта
BASE_DIR = Path(__file__).resolve().parents[2]
# Тот самый путь, куда мы сохранили модель в train_final.py
MODEL_PATH = BASE_DIR / "data/models/model.json"
DATA_PATH = BASE_DIR / "data/processed/energy_ready.csv"

def predict_local(data: pd.DataFrame):
    """Загружает модель из локального файла и делает предсказание."""
    if not MODEL_PATH.exists():
        print(f"Ошибка: Файл модели не найден по пути {MODEL_PATH}")
        return None

    # Загружаем модель XGBoost напрямую
    try:
        model = xgb.XGBRegressor()
        model.load_model(str(MODEL_PATH))
    except Exception as e:
        print(f"Ошибка при инициализации модели: {e}")
        return None

    # Очистка данных: убираем лишнее, что могло попасть из CSV
    exclude = ['timestamp', 'price']
    feature_cols = [c for c in data.columns if c not in exclude]
    
    # Важно: приводим к float64, чтобы избежать конфликтов типов в XGBoost
    X = data[feature_cols].astype(np.float64)

    try:
        predictions = model.predict(X)
        return predictions
    except Exception as e:
        print(f"Ошибка во время предсказания: {e}")
        return None

def main():
    print(f"--- Запуск инференса (локально) ---")
    
    if not DATA_PATH.exists():
        print(f"Ошибка: Файл данных {DATA_PATH} не найден.")
        return

    # 1. Загрузка данных
    df = pd.read_csv(DATA_PATH)
    
    # 2. Берем последние 5 строк для примера
    sample_data = df.tail(5)
    
    # 3. Выполняем прогноз
    results = predict_local(sample_data)

    if results is not None:
        print("\n--- Результаты прогноза (последние 5 записей) ---")
        output = pd.DataFrame({
            'timestamp': sample_data['timestamp'].values,
            'predicted_price': results
        })
        # Выводим красиво
        print(output.to_string(index=False))
        
        # Опционально: сохраняем результаты
        output_dir = BASE_DIR / "data/predictions"
        output_dir.mkdir(parents=True, exist_ok=True)
        output.to_csv(output_dir / "latest_forecast.csv", index=False)
        print(f"\nРезультаты сохранены в {output_dir}/latest_forecast.csv")

if __name__ == "__main__":
    main()