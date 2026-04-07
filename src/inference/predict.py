import pandas as pd
import numpy as np
import mlflow.sklearn
from pathlib import Path

# Конфигурация
ROOT_DIR = Path(__file__).parent.parent.parent
MLFLOW_URI = "http://localhost:5050"
MODEL_NAME = "Energy_Price_Predictor" # То имя, под которым зарегистрирована модель

def prepare_inference_data(df):
    """Подготовка одной строки данных для предсказания"""
    # Убеждаемся, что время в нужном формате
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df = df.sort_values('timestamp')

    # Те же признаки, что и при обучении
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    df['weekend'] = (df['day_of_week'] >= 5).astype(int)

    # Лаги
    for lag in [1, 2, 3, 6, 12, 24]:
        df[f'price_lag_{lag}'] = df['price'].shift(lag)

    # Скользящие окна
    df['rolling_mean_24'] = df['price'].rolling(24).mean()
    df['rolling_std_24'] = df['price'].rolling(24).std()

    # Циклические признаки
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

    # Берем только последнюю строку (самый свежий момент времени)
    last_row = df.tail(1).copy()
    
    # Список признаков должен СТРОГО совпадать с тем, что было в train_optimized.py
    feature_columns = [
        'month', 'weekend', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
        'price_lag_1', 'price_lag_2', 'price_lag_3', 'price_lag_6', 'price_lag_12', 'price_lag_24',
        'rolling_mean_24', 'rolling_std_24', 'temperature_2m', 'wind_speed_10m', 'pressure_msl'
    ]
    
    return last_row[feature_columns]

def main():
    # 1. Подключаемся к MLflow
    mlflow.set_tracking_uri(MLFLOW_URI)
    
    # 2. Загружаем модель из реестра (версия 'latest')
    print(f"Loading model '{MODEL_NAME}'...")
    try:
        model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/latest")
    except Exception as e:
        print(f"Error: Could not load model. Make sure you registered it in UI. {e}")
        return

    # 3. Читаем последние данные
    data_path = ROOT_DIR / 'data/raw/merged_energy_weather_2024.csv'
    df_raw = pd.read_csv(data_path)
    
    # 4. Готовим фичи
    X_inference = prepare_inference_data(df_raw)
    
    # 5. Делаем предсказание
    prediction = model.predict(X_inference)
    
    print("\n" + "="*30)
    print(f"PREDICTION FOR NEXT HOUR:")
    print(f"Estimated Price: {prediction[0]:.2f} EUR/MWh")
    print("="*30)

if __name__ == "__main__":
    main()