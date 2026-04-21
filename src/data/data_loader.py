import sys
import os
import time
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from entsoe import EntsoePandasClient
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Загружаем окружение из корня проекта
ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / '.env')

class EnergyDataLoader:
    """Единый класс для загрузки данных из ENTSO-E и Open-Meteo"""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv('ENTSOE_API_KEY')
        if not self.api_key:
            raise ValueError("API ключ ENTSO-E не найден в .env")
        self.entsoe_client = EntsoePandasClient(api_key=self.api_key)

    def fetch_energy_prices(self, country_code: str, start: pd.Timestamp, end: pd.Timestamp):
        """Загрузка цен на электроэнергию за указанный период"""
        print(f"Загрузка цен ENTSO-E ({country_code}) с {start.date()} по {end.date()}...")
        prices = self.entsoe_client.query_day_ahead_prices(country_code, start=start, end=end)
        df_prices = prices.reset_index()
        df_prices.columns = ['timestamp', 'price']
        return df_prices

    def fetch_weather_archive(self, lat, lon, start_date, end_date):
        """Загрузка архивной погоды с механизмом повторных попыток"""
        print(f"Загрузка погоды Open-Meteo ({lat}, {lon}) с {start_date} по {end_date}...")
        
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": lat, 
            "longitude": lon,
            "start_date": start_date, 
            "end_date": end_date,
            "hourly": [
                "temperature_2m", "relative_humidity_2m", "precipitation",
                "rain", "snowfall", "cloud_cover", "wind_speed_10m",
                "wind_direction_10m", "pressure_msl"
            ],
            "timezone": "UTC"
        }

        # Настройка сессии с повторными попытками
        session = requests.Session()
        retries = Retry(total=5, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
        session.mount('https://', HTTPAdapter(max_retries=retries))

        try:
            # Увеличиваем timeout до 60 секунд для стабильности
            response = session.get(url, params=params, timeout=60)
            response.raise_for_status()
            data = response.json()
            
            df = pd.DataFrame(data["hourly"])
            
            # Локализация через UTC
            df["timestamp"] = pd.to_datetime(df["time"]).dt.tz_localize('UTC').dt.tz_convert('Europe/Brussels')
            
            return df.drop("time", axis=1)
            
        except requests.exceptions.ConnectTimeout:
            print("Ошибка: Превышено время ожидания подключения к Open-Meteo. Проверьте интернет или VPN.")
            sys.exit(1)
        except Exception as e:
            print(f"Непредвиденная ошибка загрузки погоды: {e}")
            sys.exit(1)

    def run_full_collection(self, year: int, country: str, lat: float, lon: float, save_name: str):
        """Полный цикл: Цены + Погода за год"""
        start = pd.Timestamp(f'{year}-01-01', tz='Europe/Brussels')
        end = pd.Timestamp(f'{year}-12-31', tz='Europe/Brussels')
        
        # Загружаем цены
        prices = self.fetch_energy_prices(country, start, end)
        
        # Загружаем погоду (Open-Meteo Archive принимает строки YYYY-MM-DD)
        weather = self.fetch_weather_archive(lat, lon, f"{year}-01-01", f"{year}-12-31")
        
        if weather is None:
            print("Погода не загружена, пропускаем объединение.")
            return

        # Объединяем
        merged = prices.merge(weather, on='timestamp', how='left')
        
        # Сохранение
        save_path = ROOT_DIR / 'data' / 'raw' / save_name
        save_path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(save_path, index=False)
        if not save_path.exists():
            print("❌ Файл не был сохранен.")
            sys.exit(1)
        else:
            print(f"Данные за {year} год сохранены: {save_path}")
            return merged

if __name__ == "__main__":
    loader = EnergyDataLoader()
    
    # Параметры для Бельгии (центр Брюсселя)
    loader.run_full_collection(
        year=2024,
        country='BE',
        lat=50.8503,
        lon=4.3517,
        save_name='merged_energy_weather_2024.csv'
    )