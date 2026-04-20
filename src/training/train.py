import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import mlflow
import mlflow.sklearn
from pathlib import Path
import matplotlib.pyplot as plt
import os

# Пути проекта
ROOT_DIR = Path(__file__).parent.parent.parent
TRACKING_URI = "http://127.0.0.1:5000"

def load_and_prepare_data(file_path):
    df = pd.read_csv(file_path)
    df = df.dropna()
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['day_of_month'] = df['timestamp'].dt.day
    df['month'] = df['timestamp'].dt.month
    df['weekend'] = (df['day_of_week'] >= 5).astype(int)
    return df

def create_features(df):
    feature_columns = [
        'hour', 'day_of_week', 'day_of_month', 'month', 'weekend',
        'temperature_2m', 'relative_humidity_2m', 'precipitation', 
        'cloud_cover', 'wind_speed_10m', 'pressure_msl'
    ]
    X = df[feature_columns]
    y = df['price']
    return X, y, feature_columns

def train_model(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    return model

def setup_mlflow():
    mlflow.set_tracking_uri(TRACKING_URI)
    experiment_name = "Energy_Price_Prediction"
    
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            return mlflow.create_experiment(experiment_name)
        return experiment.experiment_id
    except Exception:
        # Если сервер все еще отдает 500 на поиск, используем дефолтный ID
        return "0"

def main():
    # --- КРИТИЧЕСКИЙ ФИКС ДЛЯ MAC ---
    # Создаем локальную папку для логов, чтобы MLflow не лез в корень системы
    local_mlruns = ROOT_DIR / "mlruns"
    local_mlruns.mkdir(exist_ok=True)
    
    print("🚀 Скрипт запущен...")
    
    # 1. Проверка связи
    mlflow.set_tracking_uri(TRACKING_URI)
    try:
        mlflow.search_experiments()
        print("✅ Связь с MLflow сервером установлена!")
    except Exception as e:
        print(f"❌ Ошибка связи с MLflow: {e}")
        return

    # 2. Данные
    data_path = ROOT_DIR / 'data/raw/merged_energy_weather_2024.csv'
    if not data_path.exists():
        print(f"📁 Файл не найден: {data_path}")
        return

    print("⏳ Загрузка и обработка данных...")
    df = load_and_prepare_data(data_path)
    X, y, feature_columns = create_features(df)
    
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    print(f"📊 Данные готовы. Обучаем на {len(X_train)} строках.")

    # 3. Обучение
    print("🧠 Обучение RandomForest...")
    model = train_model(X_train, y_train)
    
    print("🎯 Оценка модели...")
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # 4. Графики
    plot_path = ROOT_DIR / 'data/plots/prediction_comparison.png'
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(10, 5))
    plt.plot(y_test.values[:50], label='Real', color='blue', alpha=0.6)
    plt.plot(y_pred[:50], label='Pred', color='red', linestyle='--')
    plt.title("Energy Price: Real vs Predicted (First 50 samples)")
    plt.legend()
    plt.savefig(plot_path)
    plt.close()
    print("📈 График сохранен локально.")

    # 5. MLflow
    print("📡 Отправка данных в MLflow...")
    exp_id = setup_mlflow()
    
    try:
        with mlflow.start_run(experiment_id=exp_id) as run:
            # Логируем параметры
            mlflow.log_param("n_estimators", 100)
            mlflow.log_param("max_depth", 10)
            
            # Логируем метрики
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2", r2)
            
            # Важность признаков
            for name, imp in zip(feature_columns, model.feature_importances_):
                mlflow.log_metric(f"feat_{name}", imp)

            # Пытаемся сохранить артефакты
            # Мы оборачиваем это в try/except, чтобы если Mac всё же заблокирует путь, 
            # сам Run (метрики) сохранился успешно.
            try:
                mlflow.log_artifact(str(plot_path))
                # mlflow.sklearn.log_model(model, "model")
                print("📦 Артефакты успешно загружены!")
            except Exception as art_err:
                print(f"⚠️ Метрики сохранены, но артефакты не загружены: {art_err}")

            print(f"✨ Успешно! Run ID: {run.info.run_id}")
            print(f"📊 Результаты: {TRACKING_URI}")
            
    except Exception as e:
        print(f"❌ Критическая ошибка MLflow Run: {e}")

if __name__ == "__main__":
    main()