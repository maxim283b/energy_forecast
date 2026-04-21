import mlflow
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
import os
import numpy as np

# Загружаем пути и настройки
ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / '.env')

TRACKING_URI = "http://127.0.0.1:5000"
MODEL_NAME = "Energy_Forecast_XGB" 

def predict(data: pd.DataFrame, stage: str = "latest"):
    mlflow.set_tracking_uri(TRACKING_URI)
    model_uri = f"models:/{MODEL_NAME}/{stage}"
    
    print(f"Загрузка модели: {model_uri}")
    try:
        model = mlflow.pyfunc.load_model(model_uri)
    except Exception as e:
        print(f"Ошибка загрузки: {e}")
        return None

    # --- ДИНАМИЧЕСКАЯ ФИЛЬТРАЦИЯ ПРИЗНАКОВ ---
    # Получаем имена колонок, которые модель ОЖИДАЕТ (из её Signature)
    try:
        expected_columns = model.metadata.get_input_schema().input_names()
        print(f"Модель ожидает {len(expected_columns)} признаков.")
        
        # Оставляем только нужные колонки в правильном порядке
        data_filtered = data[expected_columns]
    except Exception as e:
        print(f"Ошибка приведения данных к схеме модели: {e}")
        # Если в данных физически нет того, что хочет модель (например, 'hour'),
        # тут вылетит понятная ошибка KeyError
        return None

    # Делаем предсказание на отфильтрованных данных
    predictions = model.predict(data_filtered)
    return predictions

def main():
    # Путь должен быть синхронизирован (ты использовал energy_ready.csv в обучении)
    input_path = ROOT_DIR / 'data/processed/energy_ready.csv'
    
    if not input_path.exists():
        print(f"Ошибка: Файл {input_path} не найден.")
        return

    df = pd.read_csv(input_path)
    
    # 2. Подготовка
    # Теперь нам не нужно гадать с exclude. 
    # Просто передаем весь DF (кроме целевых), а predict() сам заберет что нужно.
    X = df.drop(columns=['timestamp', 'price'], errors='ignore')

    # Приводим к float64, так как XGBoost в MLflow часто капризничает к типам
    X = X.astype(np.float64)

    # 3. Прогноз
    sample_data = X.tail(5)
    results = predict(sample_data)

    if results is not None:
        print("\n--- Результаты прогноза (последние 5 записей) ---")
        output = pd.DataFrame({
            'timestamp': df['timestamp'].tail(5),
            'predicted_price': results
        })
        print(output.to_string(index=False))

if __name__ == "__main__":
    main()