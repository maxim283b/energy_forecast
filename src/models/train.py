import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
import holidays
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import mlflow
import mlflow.xgboost
from pathlib import Path
import matplotlib.pyplot as plt
import os

# Конфигурация путей
ROOT_DIR = Path(__file__).parent.parent.parent
TRACKING_URI = "http://127.0.0.1:5000"

def load_and_prepare_data(file_path):
    """Продвинутая инженерия признаков и очистка данных"""
    df = pd.read_csv(file_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True).dt.tz_convert('Europe/Brussels')
    
    # Очистка от экстремальных выбросов (стабилизация обучения)
    upper_limit = df['price'].quantile(0.99)
    df['price'] = df['price'].clip(lower=0, upper=upper_limit)

    # Календарные признаки и праздники Бельгии
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    be_holidays = holidays.BE()
    df['is_holiday'] = df['timestamp'].apply(lambda x: 1 if x in be_holidays else 0)
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

    # Создание лаговых признаков (1 час, 24 часа, 1 неделя)
    for lag in [1, 2, 24, 48, 168]:
        df[f'price_lag_{lag}'] = df['price'].shift(lag)
    
    # Скользящие средние
    df['rolling_mean_24h'] = df['price'].rolling(window=24).mean()
    df['rolling_std_24h'] = df['price'].rolling(window=24).std()
    
    # Погодные признаки и их динамика
    df['temp_diff_24h'] = df['temperature_2m'].diff(24)
    df['wind_power_idx'] = df['wind_speed_10m'] * df['pressure_msl']

    # Циклическое кодирование времени
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

    # Удаление пустых строк, возникших из-за лагов
    return df.dropna().reset_index(drop=True)

def create_features(df):
    """Формирование признаков с исключением служебных колонок"""
    exclude = ['timestamp', 'price', 'hour', 'day_of_week']
    feature_cols = [c for c in df.columns if c not in exclude]
    
    X = df[feature_cols].astype(np.float64)
    y = df['price'].astype(np.float64)
    return X, y, feature_cols

def objective(trial, X_train, y_train, X_test, y_test):
    """Целевая функция Optuna для оптимизации гиперпараметров"""
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 800, 2000),
        'max_depth': trial.suggest_int('max_depth', 3, 7),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.05, log=True),
        'subsample': trial.suggest_float('subsample', 0.7, 0.9),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 0.9),
        'tree_method': 'hist',
        'random_state': 42
    }
    
    model = xgb.XGBRegressor(**param)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    preds = model.predict(X_test)
    return mean_absolute_error(y_test, preds)

def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    data_path = ROOT_DIR / 'data/raw/merged_energy_weather_2024.csv'
    
    print("Preparing data...")
    df = load_and_prepare_data(data_path)
    X, y, feature_cols = create_features(df)
    
    # Хронологическое разделение выборки
    split = int(len(df) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    print(f"Features count: {len(feature_cols)}")
    print("Starting hyperparameter optimization (50 trials)...")
    
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_test, y_test), n_trials=50)

    print(f"Best parameters: {study.best_params}")

    # Обучение итоговой модели
    best_model = xgb.XGBRegressor(
        **study.best_params, 
        tree_method='hist', 
        random_state=42,
        early_stopping_rounds=50
    )
    best_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    
    y_pred = best_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"Final results: MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}")

    # Логирование в MLflow
    artifact_uri = (ROOT_DIR / "mlflow_data" / "artifacts").resolve().as_uri()
    try:
        exp_name = "Energy_Price_Advanced_Final"
        exp = mlflow.get_experiment_by_name(exp_name)
        exp_id = exp.experiment_id if exp else mlflow.create_experiment(exp_name, artifact_location=artifact_uri)
        
        with mlflow.start_run(experiment_id=exp_id):
            mlflow.log_params(study.best_params)
            mlflow.log_metrics({"mae": mae, "rmse": rmse, "r2_score": r2})
            
            # Логирование важности признаков
            importances = sorted(zip(feature_cols, best_model.feature_importances_), key=lambda x: x[1], reverse=True)
            for name, imp in importances[:20]:
                mlflow.log_metric(f"imp_{name}", float(imp))

            mlflow.xgboost.log_model(best_model, "model", input_example=X_train.iloc[[0]], model_format="json")
            
            # Генерация графика
            plt.figure(figsize=(15, 7))
            plt.plot(y_test.values[-240:], label='Actual (Last 10 days)', color='blue', alpha=0.6)
            plt.plot(y_pred[-240:], label='XGBoost Prediction', color='red', linestyle='--')
            plt.title(f"Energy Price Forecast (R2: {r2:.3f})")
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plot_p = ROOT_DIR / "data/plots/final_advanced_forecast.png"
            plt.savefig(plot_p)
            plt.close()
            mlflow.log_artifact(str(plot_p))
            
            print(f"MLflow Run ID: {mlflow.active_run().info.run_id}")
            
    except Exception as e:
        print(f"MLflow logging failed: {e}")

if __name__ == "__main__":
    main()