import os
import pandas as pd
import numpy as np
import holidays
from pathlib import Path

# --- Конфигурация ---
# Используем относительные пути от корня проекта
INPUT_PATH = Path('data/raw/merged_energy_weather_2024.csv')
OUTPUT_PATH = Path('data/processed/energy_data_processed.csv')

def load_data(file_path):
    """Загрузка сырых данных и базовая проверка"""
    if not file_path.exists():
        raise FileNotFoundError(f"Файл не найден по пути: {file_path}")
    
    df = pd.read_csv(file_path)
    # Приводим к таймзоне Бельгии
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True).dt.tz_convert('Europe/Brussels')
    return df

def feature_engineering(df):
    """Генерация признаков (Feature Engineering)"""
    # 1. Обработка целевой переменной (Price)
    # Клиппинг делаем только для обучения, чтобы экстремальные пики не ломали модель
    upper_limit = df['price'].quantile(0.99)
    df['price_clipped'] = df['price'].clip(lower=0, upper=upper_limit)

    # 2. Календарные признаки
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    
    # Праздники именно для Бельгии
    be_holidays = holidays.BE()
    df['is_holiday'] = df['timestamp'].apply(lambda x: 1 if x in be_holidays else 0)
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

    # 3. Генерация лагов (Lags)
    # Важно: лаги берем от исходной цены, чтобы не было накопленной ошибки клиппинга
    for lag in [1, 2, 24, 48, 168]:
        df[f'price_lag_{lag}'] = df['price'].shift(lag)
    
    # 4. Скользящие окна (Rolling statistics)
    df['rolling_mean_24h'] = df['price'].rolling(window=24).mean()
    df['rolling_std_24h'] = df['price'].rolling(window=24).std()
    
    # 5. Погодные производные
    # Температурный градиент (как изменилась температура за сутки)
    if 'temperature_2m' in df.columns:
        df['temp_diff_24h'] = df['temperature_2m'].diff(24)
    
    # Индекс ветровой нагрузки (ветер * давление)
    if 'wind_speed_10m' in df.columns and 'pressure_msl' in df.columns:
        df['wind_power_idx'] = df['wind_speed_10m'] * df['pressure_msl']

    # 6. Циклическое кодирование времени (Sin/Cos)
    # Позволяет модели понять, что 23:00 и 00:00 — это близкие значения
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

    return df

def main():
    print("--- 🏗️ Запуск препроцессинга данных ---")
    
    try:
        # 1. Загрузка
        df = load_data(INPUT_PATH)
        print(f"Данные загружены: {len(df)} строк")

        # 2. Обработка
        df_featured = feature_engineering(df)
        
        # 3. Очистка от NaN (появляются из-за лагов и скользящих окон)
        initial_count = len(df_featured)
        df_final = df_featured.dropna().reset_index(drop=True)
        dropped_count = initial_count - len(df_final)
        print(f"🧹 Удалено строк с NaN (из-за лагов): {dropped_count}")

        # 4. Сохранение
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        
        # Сохраняем в processed. 
        # Удаляем 'price_clipped', если ты хочешь учиться на сырой цене, 
        # или оставляем её как таргет. Для XGBoost лучше использовать клиппированную.
        df_final.to_csv(OUTPUT_PATH, index=False)
        
        print(f"Файл сохранен: {OUTPUT_PATH}")
        print(f"Итоговое количество признаков: {len(df_final.columns)}")
        print(f"Охваченный период: {df_final['timestamp'].min()} --- {df_final['timestamp'].max()}")

    except Exception as e:
        print(f"Ошибка при выполнении препроцессинга: {e}")
        exit(1)

if __name__ == "__main__":
    main()