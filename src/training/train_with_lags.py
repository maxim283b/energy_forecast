import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import mlflow
import mlflow.sklearn
from pathlib import Path
import matplotlib.pyplot as plt

# Определяем корневую директорию проекта
ROOT_DIR = Path(__file__).parent.parent.parent

def load_and_prepare_data(file_path):
    """Загрузка данных, очистка и создание временных лагов"""
    print(f"⌛ Loading data from: {file_path}")
    df = pd.read_csv(file_path)

    # Приводим временную метку к нормальному виду
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    
    # Сортируем по времени (критично для лагов!)
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Календарные фичи
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    df['weekend'] = (df['day_of_week'] >= 5).astype(int)

    # Создание лагов (цена n часов назад)
    # Это дает модели "память", что критично для временных рядов
    lags = [1, 2, 3, 24, 48, 168]
    for lag in lags:
        df[f'price_lag_{lag}'] = df['price'].shift(lag)

    # Удаляем строки с NaN, которые появились из-за сдвига (первые 168 часов)
    df = df.dropna()
    return df

def create_features(df):
    """Формируем итоговый список признаков"""
    lag_columns = [f'price_lag_{lag}' for lag in [1, 2, 3, 24, 48, 168]]
    time_columns = ['hour', 'day_of_week', 'month', 'weekend']
    weather_columns = [
        'temperature_2m', 'relative_humidity_2m', 'wind_speed_10m', 
        'precipitation', 'cloud_cover', 'pressure_msl'
    ]
    
    feature_columns = time_columns + weather_columns + lag_columns
    X = df[feature_columns]
    y = df['price']
    return X, y, feature_columns

def setup_mlflow():
    """Настройка подключения к MLflow и получение ID эксперимента"""
    mlflow.set_tracking_uri("http://localhost:5050")
    experiment_name = "Energy Price Prediction with Lags"
    
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment:
        return experiment.experiment_id
    else:
        return mlflow.create_experiment(experiment_name)

def plot_predictions(y_test, y_pred, save_path):
    """Визуализация: сравнение факта и прогноза"""
    plt.figure(figsize=(15, 7))
    # Берем последние 168 часов (одну неделю) для наглядности
    plt.plot(y_test.values[-168:], label='Actual Price', color='blue', alpha=0.6)
    plt.plot(y_pred[-168:], label='Predicted Price', color='red', linestyle='--', alpha=0.8)
    plt.title('Energy Price Prediction (Last 168 hours of test set)')
    plt.xlabel('Hours')
    plt.ylabel('EUR/MWh')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path)
    plt.close()

def main():
    # 1. Путь к данным
    data_path = ROOT_DIR / 'data/raw/merged_energy_weather_2024.csv'
    if not data_path.exists():
        print(f"Error: File {data_path} not found!")
        return

    # 2. Подготовка
    df = load_and_prepare_data(data_path)
    X, y, feature_columns = create_features(df)

    # 3. Хронологический сплит 
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"Data ready. Features: {len(feature_columns)}")
    print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples...")

    # 4. Обучение
    model = RandomForestRegressor(
        n_estimators=200, 
        max_depth=15, 
        random_state=42, 
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # 5. Оценка
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"\n🚀 RESULTS:")
    print(f"R2 Score: {r2:.4f}")
    print(f"MAE: {mae:.4f} EUR/MWh")

    # 6. График
    plot_path = ROOT_DIR / 'data/plots/prediction_with_lags.png'
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plot_predictions(y_test, y_pred, plot_path)

    # 7. Логирование в MLflow
    experiment_id = setup_mlflow()
    with mlflow.start_run(experiment_id=experiment_id):
        # Логируем параметры
        mlflow.log_params({
            "model_type": "RandomForest",
            "n_estimators": 200,
            "max_depth": 15,
            "use_lags": True,
            "use_weather": True
        })

        # Логируем метрики
        mlflow.log_metrics({
            "r2_score": r2,
            "mae": mae,
            "rmse": rmse
        })

        # Логируем важность признаков
        importances = model.feature_importances_
        for name, imp in zip(feature_columns, importances):
            mlflow.log_metric(f"importance_{name}", imp)

        # Сохраняем модель и график
        mlflow.sklearn.log_model(model, "rf_lags_weather_model")
        mlflow.log_artifact(plot_path)
        
        print(f"\n Run logged to MLflow (ID: {mlflow.active_run().info.run_id})")

if __name__ == "__main__":
    main()