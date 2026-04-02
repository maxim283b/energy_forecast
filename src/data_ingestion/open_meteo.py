import requests
import pandas as pd

def fetch_weather(latitude, longitude, start_date, end_date):
    """Получить исторические данные погоды"""
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ["temperature_2m", "precipitation", "wind_speed_10m"]
    }
    response = requests.get(url, params=params)
    return pd.DataFrame(response.json()["hourly"])