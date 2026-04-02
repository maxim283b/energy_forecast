import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import mlflow
from pathlib import Path
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).parent.parent.parent


def load_and_prepare_data(file_path):
    """Загрузить и подготовить данные с погодными признаками"""
    df = pd.read_csv(file_path)

    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df['timestamp'] = df['timestamp'].dt.tz_convert('Europe/Brussels')

    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['day_of_month'] = df['timestamp'].dt.day
    df['month'] = df['timestamp'].dt.month
    df['weekend'] = (df['day_of_week'] >= 5).astype(int)

    for lag in [1, 2, 3, 6, 12, 24, 48, 72, 168]:
        df[f'price_lag_{lag}'] = df['price'].shift(lag)

    df = df.fillna(method='bfill').fillna(method='ffill')

    return df


def create_features(df):
    """Создать признаки для модели включая погоду"""
    lag_columns = [f'price_lag_{lag}' for lag in [1, 2, 3, 6, 12, 24, 48, 72, 168]]
    time_columns = ['hour', 'day_of_week', 'day_of_month', 'month', 'weekend']

    weather_columns = [
        'temperature_2m',
        'relative_humidity_2m',
        'precipitation',
        'rain',
        'snowfall',
        'cloud_cover',
        'wind_speed_10m',
        'wind_direction_10m',
        'pressure_msl'
    ]

    existing_weather = [col for col in weather_columns if col in df.columns]

    feature_columns = time_columns + lag_columns + existing_weather

    X = df[feature_columns]
    y = df['price']

    return X, y, feature_columns


def train_model(X_train, y_train, n_estimators=200, max_depth=15):
    """Обучить модель RandomForest"""
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """Оценить качество модели"""
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    return mae, rmse, r2, y_pred


def plot_predictions(y_test, y_pred, save_path):
    """Построить график предсказаний"""
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.values[:168], label='Actual', alpha=0.7)
    plt.plot(y_pred[:168], label='Predicted', alpha=0.7)
    plt.xlabel('Hour')
    plt.ylabel('Price (EUR/MWh)')
    plt.title('Energy Price Prediction with Weather Features')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path)
    plt.close()


def setup_mlflow():
    """Настроить MLflow"""
    mlflow_dir = ROOT_DIR / 'mlflow'
    mlflow_dir.mkdir(exist_ok=True)
    mlflow.set_tracking_uri(f"file://{mlflow_dir}")

    experiment_name = "Energy Price Prediction with Weather"
    experiment = mlflow.get_experiment_by_name(experiment_name)

    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
        print(f"Created experiment: {experiment_name}")
    else:
        experiment_id = experiment.experiment_id
        print(f"Using existing experiment: {experiment_name}")

    return experiment_id


def main():
    """Основная функция обучения с погодными признаками"""
    data_path = ROOT_DIR / 'data/raw/merged_energy_weather_2024.csv'

    if not data_path.exists():
        print(f"Data file not found: {data_path}")
        print("Please run entsoe_parser.py first")
        return

    print("Loading and preparing data...")
    df = load_and_prepare_data(data_path)
    print(f"Data loaded: {len(df)} records")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

    X, y, feature_columns = create_features(df)
    print(f"Total features: {len(feature_columns)}")

    time_features = [f for f in feature_columns if f in ['hour', 'day_of_week', 'day_of_month', 'month', 'weekend']]
    lag_features = [f for f in feature_columns if f.startswith('price_lag_')]
    weather_features = [f for f in feature_columns if f not in time_features + lag_features]

    print(f"  Time features: {len(time_features)}")
    print(f"  Lag features: {len(lag_features)}")
    print(f"  Weather features: {len(weather_features)}")

    split_idx = int(len(df) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"Train set: {len(X_train)} records")
    print(f"Test set: {len(X_test)} records")

    print("\nTraining model...")
    model = train_model(X_train, y_train)

    mae, rmse, r2, y_pred = evaluate_model(model, X_test, y_test)

    print("\n" + "=" * 50)
    print("RESULTS WITH WEATHER FEATURES")
    print("=" * 50)
    print(f"MAE: {mae:.4f} EUR/MWh")
    print(f"RMSE: {rmse:.4f} EUR/MWh")
    print(f"R2: {r2:.4f}")

    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\nTop 15 Feature Importance:")
    for _, row in feature_importance.head(15).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")

    plot_path = ROOT_DIR / 'data/plots/prediction_with_weather.png'
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plot_predictions(y_test, y_pred, plot_path)
    print(f"\nPlot saved to {plot_path}")

    experiment_id = setup_mlflow()

    with mlflow.start_run(experiment_id=experiment_id):
        mlflow.log_param("n_estimators", 200)
        mlflow.log_param("max_depth", 15)
        mlflow.log_param("has_lags", True)
        mlflow.log_param("has_weather", True)
        mlflow.log_param("weather_features", len(weather_features))

        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2_score", r2)

        for _, row in feature_importance.head(15).iterrows():
            mlflow.log_metric(f"importance_{row['feature']}", row['importance'])

        mlflow.sklearn.log_model(model, "random_forest_with_weather")
        mlflow.log_artifact(plot_path)

        print(f"\nExperiment saved to MLflow")
        print(f"Run ID: {mlflow.active_run().info.run_id}")

    print("\nTraining completed")


if __name__ == "__main__":
    main()