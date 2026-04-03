import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import mlflow
from pathlib import Path
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).parent.parent.parent

def load_and_prepare_data(file_path):
    """Загрузка и адаптация данных Open-Meteo"""
    df = pd.read_csv(file_path)

    # Адаптация колонок под логику модели
    if 'temperature' in df.columns:
        df = df.rename(columns={'temperature': 'price'})
    
    # Конвертация времени (Open-Meteo формат)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Генерация признаков
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['day_of_month'] = df['timestamp'].dt.day
    df['month'] = df['timestamp'].dt.month
    df['weekend'] = (df['day_of_week'] >= 5).astype(int)

    return df

def create_features(df):
    feature_columns = ['hour', 'day_of_week', 'day_of_month', 'month', 'weekend']
    X = df[feature_columns]
    y = df['price']
    return X, y, feature_columns

def train_model(X_train, y_train, n_estimators=100, max_depth=10):
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    return mae, rmse, r2, y_pred

def plot_predictions(y_test, y_pred, save_path):
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.values[:100], label='Actual', alpha=0.7)
    plt.plot(y_pred[:100], label='Predicted', alpha=0.7)
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.title('Weather/Energy Prediction')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path)
    plt.close()

def setup_mlflow():
    # Настройка на удаленный сервер в Docker
    mlflow.set_tracking_uri("http://localhost:5050") 
    
    experiment_name = "Energy Price Prediction"
    experiment = mlflow.get_experiment_by_name(experiment_name)

    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id

    return experiment_id


def main():
    data_path = ROOT_DIR / 'data/raw/weather_data.csv'
    
    print(f"DEBUG: Looking for data at {data_path}") # Добавь это
    
    if not data_path.exists():
        print(f"❌ Data file not found at: {data_path}")
        print(f"Current working directory: {Path.cwd()}")
        return

    print("✅ File found. Loading data...")
    df = load_and_prepare_data(data_path)
    
    print(f"✅ Data loaded. Rows: {len(df)}")
    X, y, feature_columns = create_features(df)

    split_idx = int(len(df) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    model = train_model(X_train, y_train)
    mae, rmse, r2, y_pred = evaluate_model(model, X_test, y_test)

    # Подготовка путей для артефактов
    plot_path = ROOT_DIR / 'data/plots/prediction_comparison.png'
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plot_predictions(y_test, y_pred, plot_path)

    experiment_id = setup_mlflow()

    with mlflow.start_run(experiment_id=experiment_id):
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 10)
        
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2_score", r2)

        # Логирование важности признаков
        for name, imp in zip(feature_columns, model.feature_importances_):
            mlflow.log_metric(f"importance_{name}", imp)

        mlflow.sklearn.log_model(model, "random_forest_model")
        mlflow.log_artifact(plot_path)

        print(f"Run ID: {mlflow.active_run().info.run_id}")

    print("Training completed successfully")

if __name__ == "__main__":
    main()