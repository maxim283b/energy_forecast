# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from entsoe import EntsoePandasClient
import os
import requests
from dotenv import load_dotenv
from pathlib import Path
import time

ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / '.env')

class EnergyDataGoldMiner:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv('ENTSOE_API_KEY')
        if not self.api_key:
            raise ValueError("API ключ ENTSO-E не найден")
        self.client = EntsoePandasClient(api_key=self.api_key)

    def fetch_year_data(self, country_code: str, lat: float, lon: float, year: int):
        country_code = country_code.upper()
        start = pd.Timestamp(f'{year}-01-01', tz='Europe/Brussels')
        end = pd.Timestamp(f'{year}-12-31 23:00:00', tz='Europe/Brussels')

        print(f"\n--- Mining data for {year} ({country_code}) ---")

        # 1. Цены целевой страны (Target)
        try:
            prices = self.client.query_day_ahead_prices(country_code, start=start, end=end)
            df = prices.reset_index()
            df.columns = ['timestamp', 'price']
            df = df.drop_duplicates(subset=['timestamp'])
        except Exception as e:
            print(f"Critical error fetching prices for {country_code}: {e}")
            return None

        # 2. Цены соседних стран (FR, DE, NL) - Ключ к 0.75+
        neighbors = ['FR', 'DE_LU', 'NL']
        for neighbor in neighbors:
            try:
                print(f"  Fetching neighbor prices: {neighbor}...")
                n_prices = self.client.query_day_ahead_prices(neighbor, start=start, end=end)
                n_df = n_prices.reset_index()
                n_df.columns = ['timestamp', f'price_{neighbor.split("_")[0].lower()}']
                n_df = n_df.drop_duplicates(subset=['timestamp'])
                df = df.merge(n_df, on='timestamp', how='left')
            except Exception as e:
                print(f"  Could not fetch prices for {neighbor}: {e}")

        # 3. Прогноз нагрузки (Load Forecast)
        try:
            load_f = self.client.query_load_forecast(country_code, start=start, end=end)
            load_df = load_f.reset_index()
            load_df.columns = ['timestamp', 'load_forecast']
            df = df.merge(load_df.drop_duplicates(subset=['timestamp']), on='timestamp', how='left')
            print("  Load forecast added")
        except Exception as e:
            print(f"  Skip load forecast: {e}")

        # 4. Прогноз генерации (Solar/Wind)
        try:
            gen_f = self.client.query_wind_and_solar_forecast(country_code, start=start, end=end)
            gen_df = gen_f.reset_index()
            rename_dict = {'index': 'timestamp'}
            for col in gen_df.columns:
                if 'Solar' in col: rename_dict[col] = 'solar_forecast'
                if 'Wind' in col: rename_dict[col] = 'wind_forecast'
            gen_df = gen_df.rename(columns=rename_dict)
            
            wind_cols = [c for c in gen_df.columns if 'wind_forecast' in c]
            if len(wind_cols) > 1:
                gen_df['wind_forecast'] = gen_df[wind_cols].sum(axis=1)
            
            cols = ['timestamp'] + [c for c in ['solar_forecast', 'wind_forecast'] if c in gen_df.columns]
            df = df.merge(gen_df[cols].drop_duplicates(subset=['timestamp']), on='timestamp', how='left')
            print("  Generation forecast added")
        except Exception as e:
            print(f"  Skip generation forecast: {e}")

        # 5. Погода
        weather = self._fetch_weather(lat, lon, year)
        if weather is not None:
            df = df.merge(weather, on='timestamp', how='left')
            print("  Weather data added")

        return df

    def _fetch_weather(self, lat, lon, year):
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": lat, "longitude": lon,
            "start_date": f"{year}-01-01", "end_date": f"{year}-12-31",
            "hourly": ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "direct_radiation"],
            "timezone": "UTC"
        }
        try:
            response = requests.get(url, params=params, timeout=30)
            w_df = pd.DataFrame(response.json()["hourly"])
            w_df["timestamp"] = pd.to_datetime(w_df["time"], utc=True).dt.tz_convert('Europe/Brussels')
            return w_df.drop("time", axis=1)
        except Exception as e:
            print(f"  Weather error: {e}")
            return None

def main():
    miner = EnergyDataGoldMiner()
    years = [2022, 2023, 2024]
    all_data = []

    for y in years:
        res = miner.fetch_year_data('BE', 50.85, 4.35, y)
        if res is not None: all_data.append(res)
        time.sleep(2)

    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        final_df = final_df.sort_values('timestamp').reset_index(drop=True)
        save_path = ROOT_DIR / 'data/raw/big_energy_dataset_v2.csv'
        final_df.to_csv(save_path, index=False)
        print(f"\nГотово! Колонки: {final_df.columns.tolist()}")

if __name__ == "__main__":
    main()