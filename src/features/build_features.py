# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import holidays
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
INPUT_PATH = BASE_DIR / 'data/raw/big_energy_dataset_v2.csv'
OUTPUT_PATH = BASE_DIR / 'data/processed/energy_ready.csv'

def add_time_features(df):
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True).dt.tz_convert('Europe/Brussels')
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    
    be_holidays = holidays.BE()
    df['is_holiday'] = df['timestamp'].dt.date.apply(lambda x: 1 if x in be_holidays else 0)
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    return df

def add_market_physics_features(df):
    # 1. Заполнение пропусков в прогнозах
    cols_to_fix = ['solar_forecast', 'wind_forecast', 'load_forecast']
    for col in cols_to_fix:
        if col in df.columns:
            df[col] = df[col].ffill().fillna(0)

    # 2. Net Load (Критический признак для R2)
    df['net_load_forecast'] = df['load_forecast'] - df['solar_forecast'] - df['wind_forecast']
    
    # 3. Энергетический микс
    df['renewable_total'] = df['solar_forecast'] + df['wind_forecast']
    df['non_renewable_needed'] = df['load_forecast'] - df['renewable_total']
    
    return df

def add_neighbor_features(df):
    """
    Добавление влияния соседних рынков. 
    Важно: используем лаг 24, так как мы знаем только вчерашние цены соседей.
    """
    neighbors = ['fr', 'de', 'nl']
    for n in neighbors:
        col = f'price_{n}'
        if col in df.columns:
            # Лаг 24: вчерашняя цена соседа в этот же час
            df[f'{col}_lag_24'] = df[col].shift(24)
            # Спред: разница цен между BE и соседом вчера
            # Если спред большой, значит вчера был активный экспорт/импорт
            df[f'spread_be_{n}_lag_24'] = df['price'].shift(24) - df[f'{col}_lag_24']
            
    return df

def add_lags_and_rolling(df):
    # Лаги целевой переменной (BE price)
    for lag in [24, 48, 168]:
        df[f'price_lag_{lag}'] = df['price'].shift(lag)
    
    # Скользящие показатели
    df['price_mean_24h'] = df['price'].shift(24).rolling(window=24).mean()
    df['price_std_24h'] = df['price'].shift(24).rolling(window=24).std()
    
    # Динамика нагрузки
    df['load_trend_24h'] = df['load_forecast'] - df['load_forecast'].shift(24)
    
    return df

def main():
    print("--- Start Feature Engineering: Cross-Border & Physics Protocol ---")
    if not INPUT_PATH.exists():
        print(f"Error: {INPUT_PATH} not found")
        return

    df = pd.read_csv(INPUT_PATH)
    
    # Ограничение выбросов (Clipping)
    upper_limit = df['price'].quantile(0.99)
    df['target'] = df['price'].clip(lower=-50, upper=upper_limit)
    
    # Генерация признаков
    df = add_time_features(df)
    df = add_market_physics_features(df)
    df = add_neighbor_features(df)
    df = add_lags_and_rolling(df)
    
    # Формируем итоговый список фичей
    features = [
        'target', 'timestamp',
        # Время
        'hour_sin', 'hour_cos', 'day_of_week', 'is_holiday', 'is_weekend',
        # Бельгия: Прогнозы (спрос/предложение)
        'load_forecast', 'net_load_forecast', 'solar_forecast', 'wind_forecast',
        'renewable_total', 'non_renewable_needed', 'load_trend_24h',
        # Соседи (Вчерашний контекст)
        'price_fr_lag_24', 'price_de_lag_24', 'price_nl_lag_24',
        'spread_be_fr_lag_24', 'spread_be_de_lag_24', 'spread_be_nl_lag_24',
        # Погода
        'temperature_2m', 'wind_speed_10m', 'direct_radiation',
        # История цен
        'price_lag_24', 'price_lag_48', 'price_lag_168', 'price_mean_24h', 'price_std_24h'
    ]
    
    final_df = df[[c for c in features if c in df.columns]].dropna().reset_index(drop=True)
    
    print(f"Total features: {len(final_df.columns) - 2}")
    print(f"Final shape: {final_df.shape}")
    
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Success. Ready for training.")

if __name__ == "__main__":
    main()