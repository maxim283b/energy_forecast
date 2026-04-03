import requests
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

def fetch_and_save_weather(lat, lon, days_back=10):
    """Скачивает погоду за последние N дней и сохраняет в data/raw"""
    
    # Определяем даты (Open-Meteo Archive требует формат YYYY-MM-DD)
    end_date = (datetime.now() - timedelta(days=2)).strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
    
    print(f"Fetching weather from {start_date} to {end_date}...")
    
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ["temperature_2m", "relative_humidity_2m", "wind_speed_10m"],
        "timezone": "Europe/Berlin"
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        # Превращаем в таблицу
        df = pd.DataFrame(data["hourly"])
        df = df.rename(columns={
            "time": "timestamp",
            "temperature_2m": "temperature",
            "relative_humidity_2m": "humidity"
        })
        
        # Определяем путь сохранения
        root_dir = Path(__file__).parent.parent.parent
        save_path = root_dir / 'data/raw/weather_data.csv'
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Сохраняем
        df.to_csv(save_path, index=False)
        print(f"Saved {len(df)} rows to {save_path}")
        return save_path

    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    # Координаты (например, Амстердам, раз вы работаете с NL)
    fetch_and_save_weather(lat=52.3676, lon=4.9041, days_back=30)