import pandas as pd
import logging
import os

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class EnergyDataLoader:
    def __init__(self, raw_data_dir="data/raw"):
        self.raw_data_dir = raw_data_dir
        os.makedirs(self.raw_data_dir, exist_ok=True)

    def load_sample_data(self):
        """Создает тестовые данные, пока нет реальных"""
        logging.info("Generating sample energy data...")
        data = {
            'timestamp': pd.date_range(start='2024-01-01', periods=100, freq='H'),
            'consumption': [x * 1.5 for x in range(100)],
            'temperature': [20 + (x % 10) for x in range(100)]
        }
        df = pd.DataFrame(data)
        return df

    def save_data(self, df, filename="energy_raw.csv"):
        """Сохраняет данные в папку, за которой будет следить DVC"""
        path = os.path.join(self.raw_data_dir, filename)
        df.to_csv(path, index=False)
        logging.info(f"Data saved to {path}. Ready for DVC tracking!")

if __name__ == "__main__":
    loader = EnergyDataLoader()
    sample_df = loader.load_sample_data()
    loader.save_data(sample_df)