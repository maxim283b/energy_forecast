import pandas as pd
import numpy as np
import holidays
from pathlib import Path

def add_time_features(df):
    """Календарные и циклические признаки"""
    # Гарантируем, что работаем с объектами datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    
    # Праздники Бельгии
    be_holidays = holidays.BE()
    # Преобразуем дату для сравнения с календарем праздников
    df['is_holiday'] = df['timestamp'].dt.date.apply(lambda x: 1 if x in be_holidays else 0)
    
    # Циклы
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    return df

def add_lags(df, lags=[1, 2, 24, 48, 168]):
    """Генерация лагов для целевой переменной price"""
    for lag in lags:
        df[f'price_lag_{lag}'] = df['price'].shift(lag)
    return df

def main():
    # Путь должен соответствовать выходу make_dataset.py
    input_path = Path('data/interim/energy_cleaned.csv')
    output_path = Path('data/processed/energy_ready.csv')

    if not input_path.exists():
        print(f"Файл {input_path} не найден")
        return

    # Чтение данных
    input_df = pd.read_csv(input_path)
    
    # Преобразование в datetime с utc=True решает проблему смешанных смещений (DST)
    input_df['timestamp'] = pd.to_datetime(input_df['timestamp'], utc=True)
    
    # Добавление признаков
    df = add_time_features(input_df)
    df = add_lags(df)
    
    # Сохранение финального результата
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Удаляем строки с NaN (появившиеся из-за лагов)
    df.dropna().to_csv(output_path, index=False)
    print(f"Финальные признаки сохранены в {output_path}")

if __name__ == "__main__":
    main()