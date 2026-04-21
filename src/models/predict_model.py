# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import xgboost as xgb
from pathlib import Path

# Определение путей
BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = BASE_DIR / "data/models/model.json" 
DATA_PATH = BASE_DIR / "data/processed/energy_ready.csv"

def predict_local(data):
    """Выполнение прогноза на локальной модели"""
    # 1. Загрузка модели
    model = xgb.XGBRegressor()
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Модель не найдена по пути: {MODEL_PATH}")
    model.load_model(str(MODEL_PATH))
    
    # 2. Получаем список признаков из модели
    model_features = model.get_booster().feature_names
    
    # 3. Фильтруем данные: берем только те колонки, которые ЕСТЬ и в модели, и в данных
    # Это предотвратит падение, если в данных чуть больше или чуть меньше колонок
    available_features = [f for f in model_features if f in data.columns]
    
    if len(available_features) < len(model_features):
        missing = set(model_features) - set(available_features)
        print(f"Внимание: В данных отсутствуют признаки {missing}. Заполняем нулями.")
        for col in missing:
            data[col] = 0
            
    # 4. Выстраиваем колонки в СТРОГОМ порядке, как ждет XGBoost
    X = data[model_features].astype(np.float64)
    
    # 5. Прогноз
    preds = model.predict(X)
    return preds

def main():
    print(f"--- 🚀 Запуск инференса (проверка пайплайна) ---")
    
    if not DATA_PATH.exists():
        print(f"Ошибка: Файл данных {DATA_PATH} не найден.")
        return

    # 1. Загрузка данных
    df = pd.read_csv(DATA_PATH)
    
    # 2. Берем последние 5 строк для примера
    sample_data = df.tail(5).copy() # Используем copy(), чтобы не было SettingWithCopyWarning
    
    try:
        # 3. Выполняем прогноз
        results = predict_local(sample_data)

        print("\n--- Результаты прогноза (последние 5 часов) ---")
        output = pd.DataFrame({
            'timestamp': sample_data['timestamp'].values,
            'predicted_price': results
        })
        print(output.to_string(index=False))
        
        output_dir = BASE_DIR / "data/predictions"
        output_dir.mkdir(parents=True, exist_ok=True)
        output.to_csv(output_dir / "latest_forecast.csv", index=False)
        print(f"\nПрогноз успешно сохранен в {output_dir}/latest_forecast.csv")
        
    except Exception as e:
        print(f"Ошибка при выполнении инференса: {e}")

if __name__ == "__main__":
    main()