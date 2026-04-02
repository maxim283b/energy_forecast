import pandas as pd
import numpy as np
from entsoe import EntsoePandasClient
import os
import requests
from dotenv import load_dotenv
from pathlib import Path
import time

ROOT_DIR = Path(__file__).parent.parent.parent
load_dotenv(ROOT_DIR / '.env')


class EntsoeClient:
    """Клиент для работы с API ENTSO-E и Open-Meteo"""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv('ENTSOE_API_KEY')
        if not self.api_key:
            raise ValueError("API ключ ENTSO-E не найден")
        self.client = EntsoePandasClient(api_key=self.api_key)

    def get_day_ahead_prices(self, country_code: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        """Получить дневные цены на электроэнергию"""
        return self.client.query_day_ahead_prices(country_code, start=start, end=end)

    def fetch_weather_by_month(self, latitude: float, longitude: float, year: int, month: int):
        """Получить погодные данные за месяц"""
        start_date = f"{year}-{month:02d}-01"

        if month == 12:
            end_date = f"{year}-12-31"
        else:
            end_date = f"{year}-{month + 1:02d}-01"

        url = "https://archive-api.open-meteo.com/v1/archive"

        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": [
                "temperature_2m",
                "relative_humidity_2m",
                "precipitation",
                "rain",
                "snowfall",
                "cloud_cover",
                "wind_speed_10m",
                "wind_direction_10m",
                "pressure_msl"
            ],
            "timezone": "Europe/Brussels"
        }

        print(f"  Fetching {year}-{month:02d}...")
        try:
            response = requests.get(url, params=params, timeout=30)
            data = response.json()

            df = pd.DataFrame(data["hourly"])
            df["timestamp"] = pd.to_datetime(df["time"], utc=False)
            df = df.drop("time", axis=1)

            df["timestamp"] = df["timestamp"].dt.tz_localize('Europe/Brussels', ambiguous='NaT', nonexistent='NaT')
            df = df.dropna(subset=['timestamp'])

            print(f"    Retrieved {len(df)} records")
            return df
        except Exception as e:
            print(f"    Error for month {month}: {e}")
            return None

    def fetch_weather_data(self, latitude: float, longitude: float, year: int, save_path: str = None):
        """Загрузить погодные данные по месяцам"""
        all_data = []

        for month in range(1, 13):
            df_month = self.fetch_weather_by_month(latitude, longitude, year, month)
            if df_month is not None and len(df_month) > 0:
                all_data.append(df_month)
            time.sleep(0.5)

        if all_data:
            weather = pd.concat(all_data, ignore_index=True)
            weather = weather.sort_values('timestamp').reset_index(drop=True)
        else:
            print("No weather data retrieved, creating synthetic data")
            weather = None

        if save_path and weather is not None:
            full_path = ROOT_DIR / save_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            weather.to_csv(full_path, index=False)
            print(f"Weather data saved to {full_path}")

        return weather

    def fetch_energy_prices(self, country_code: str, year: int, save_path: str = None):
        """Загрузить цены на электроэнергию за год"""
        start = pd.Timestamp(f'{year}-01-01', tz='Europe/Brussels')
        end = pd.Timestamp(f'{year}-12-31', tz='Europe/Brussels')

        print(f"Fetching energy prices for {year}...")
        prices = self.get_day_ahead_prices(country_code, start, end)

        df_prices = prices.reset_index()
        df_prices.columns = ['timestamp', 'price']

        if save_path:
            full_path = ROOT_DIR / save_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            df_prices.to_csv(full_path, index=False)
            print(f"Energy prices saved to {full_path}")

        return df_prices

    def create_synthetic_weather(self, prices_df):
        """Создать синтетические погодные данные для тестирования"""
        np.random.seed(42)

        weather = pd.DataFrame()
        weather['timestamp'] = prices_df['timestamp'].copy()

        weather['temperature_2m'] = (
                10 +
                10 * np.sin(2 * np.pi * weather['timestamp'].dt.dayofyear / 365) +
                np.random.normal(0, 5, len(weather))
        )

        weather['relative_humidity_2m'] = (
                70 +
                20 * np.sin(2 * np.pi * weather['timestamp'].dt.hour / 24) +
                np.random.normal(0, 10, len(weather))
        )

        weather['precipitation'] = np.random.exponential(0.5, len(weather))
        weather['rain'] = np.random.exponential(0.3, len(weather))
        weather['snowfall'] = np.random.exponential(0.1, len(weather))
        weather['cloud_cover'] = 50 + 30 * np.random.random(len(weather))
        weather['wind_speed_10m'] = 5 + 3 * np.random.random(len(weather))
        weather['wind_direction_10m'] = np.random.uniform(0, 360, len(weather))
        weather['pressure_msl'] = 1013 + 10 * np.random.random(len(weather))

        return weather

    def fetch_and_merge(self, country_code: str, latitude: float, longitude: float,
                        year: int, save_path: str = None):
        """Загрузить и объединить цены с погодой"""
        prices = self.fetch_energy_prices(country_code, year)

        weather = self.fetch_weather_data(latitude, longitude, year)

        if weather is None or len(weather) == 0:
            print("\nCreating synthetic weather data...")
            weather = self.create_synthetic_weather(prices)

        weather['timestamp'] = pd.to_datetime(weather['timestamp'])
        weather = weather.sort_values('timestamp')

        merged = prices.merge(weather, on='timestamp', how='left')

        if save_path:
            full_path = ROOT_DIR / save_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            merged.to_csv(full_path, index=False)
            print(f"Merged data saved to {full_path}")

        return merged


if __name__ == "__main__":
    client = EntsoeClient()

    print("Loading data for 2024...")
    print("-" * 50)

    merged = client.fetch_and_merge(
        country_code='NL',
        latitude=52.37,
        longitude=4.89,
        year=2024,
        save_path='data/raw/merged_energy_weather_2024.csv'
    )

    print("\n" + "=" * 50)
    print("Data loaded successfully")
    print(f"Total records: {len(merged)}")
    print(f"Columns: {merged.columns.tolist()}")

    if 'timestamp' in merged.columns:
        print(f"Date range: {merged['timestamp'].min()} to {merged['timestamp'].max()}")

    print(f"\nFirst 5 rows:")
    print(merged.head())
    print(f"\nMissing values:")
    print(merged.isnull().sum())