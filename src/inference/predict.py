import pandas as pd
import numpy as np
import mlflow.xgboost
import holidays
from pathlib import Path

# Конфигурация
ROOT_DIR = Path(__file__).parent.parent.parent
MLFLOW_URI = "http://localhost:5000"
# Используйте Run ID из последнего лога: ae096b4827084a2ab3c52ae843392129
RUN_ID = "ae096b4827084a2ab3c52ae843392129" 

def prepare_inference_data(df):
    """Подготовка данных с автоматическим выбором признаков"""
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True).dt.tz_convert('Europe/Brussels')
    df = df.sort_values('timestamp')

    # 1. Календарь и праздники
    be_holidays = holidays.BE()
    df['is_holiday'] = df['timestamp'].apply(lambda x: 1 if x in be_holidays else 0)
    df['is_weekend'] = (df['timestamp'].dt.dayofweek >= 5).astype(int)
    df['month'] = df['timestamp'].dt.month

    # 2. Лаги
    for lag in [1, 2, 24, 48, 168]:
        df[f'price_lag_{lag}'] = df['price'].shift(lag)

    # 3. Скользящие окна
    df['rolling_mean_24h'] = df['price'].rolling(window=24).mean()
    df['rolling_std_24h'] = df['price'].rolling(window=24).std()
    
    # 4. Погодные взаимодействия
    df['temp_diff_24h'] = df['temperature_2m'].diff(24)
    df['wind_power_idx'] = df['wind_speed_10m'] * df['pressure_msl']

    # 5. Циклические признаки
    hour = df['timestamp'].dt.hour
    day = df['timestamp'].dt.dayofweek
    df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * hour / 24)
    df['day_sin'] = np.sin(2 * np.pi * day / 7)
    df['day_cos'] = np.cos(2 * np.pi * day / 7)

    # Автоматическое определение колонок (как в train_final.py)
    exclude = ['timestamp', 'price', 'hour', 'day_of_week']
    feature_columns = [c for c in df.columns if c not in exclude]
    
    last_row = df.tail(1).copy()
    return last_row[feature_columns].astype(np.float64)

def main():
    mlflow.set_tracking_uri(MLFLOW_URI)
    
    # Загружаем модель
    model_uri = f"runs:/{RUN_ID}/model"
    print(f"Loading XGBoost model from run {RUN_ID}...")
    
    model = mlflow.xgboost.load_model(model_uri)

    # Получаем имена признаков напрямую из модели, чтобы не гадать
    # Это самый надежный способ избежать KeyError
    try:
        expected_features = model.get_booster().feature_names
    except:
        expected_features = None

    data_path = ROOT_DIR / 'data/raw/merged_energy_weather_2024.csv'
    df_raw = pd.read_csv(data_path)
    
    X_inference = prepare_inference_data(df_raw)
    
    # Если модель ожидает конкретный порядок признаков
    if expected_features:
        X_inference = X_inference[expected_features]
    
    prediction = model.predict(X_inference)
    
    print("\n" + "="*40)
    print(f"ENERGY PRICE PREDICTION:")
    print(f"Forecast for the next available hour")
    print(f"Estimated Price: {prediction[0]:.2f} EUR/MWh")
    print("="*40)

if __name__ == "__main__":
    main()