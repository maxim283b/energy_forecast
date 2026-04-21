import os
import mlflow
import mlflow.xgboost
import optuna
import pandas as pd
import numpy as np
import xgboost as xgb
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from mlflow.models.signature import infer_signature

# --- Конфигурация ---
DATA_PATH = Path('data/processed/energy_ready.csv')
TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "Energy_Forecast_Production"

def load_processed_data(file_path):
    if not file_path.exists():
        raise FileNotFoundError(f"Файл не найден: {file_path}")
    df = pd.read_csv(file_path)
    exclude = ['timestamp', 'price']
    feature_cols = [c for c in df.columns if c not in exclude]
    X = df[feature_cols].astype(np.float64)
    y = df['price'].astype(np.float64)
    return X, y, feature_cols

def objective(trial, X_train, y_train, X_test, y_test):
    # ВАЖНО: используем nested=True, чтобы не смешивать параметры триалов с главным запуском
    with mlflow.start_run(nested=True):
        param = {
            'n_estimators': trial.suggest_int('n_estimators', 800, 1500),
            'max_depth': trial.suggest_int('max_depth', 3, 6),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.05, log=True),
            'subsample': trial.suggest_float('subsample', 0.7, 0.9),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 0.9),
            'tree_method': 'hist',
            'random_state': 42
        }
        
        # Чтобы автологирование не конфликтовало внутри Optuna, 
        # можно временно отключить его для триалов или просто логировать метрику вручную
        model = xgb.XGBRegressor(**param)
        model.fit(X_train, y_train)
        
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        
        mlflow.log_params(param)
        mlflow.log_metric("trial_mae", mae)
        return mae

def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    # Включаем автологирование, но настраиваем его так, 
    # чтобы оно не мешало ручному логированию в цикле Optuna
    mlflow.xgboost.autolog(importance_types=['gain'], log_models=False, silent=True)

    print("--- 1. Загрузка подготовленных данных ---")
    X, y, feature_cols = load_processed_data(DATA_PATH)
    
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Один главный запуск для всего процесса
    with mlflow.start_run(run_name="Final_Model_Training") as run:
        
        print("--- 2. Поиск гиперпараметров (Optuna) ---")
        # Отключаем автологирование на время поиска, чтобы избежать INVALID_PARAMETER_VALUE
        mlflow.xgboost.autolog(disable=True)
        
        study = optuna.create_study(direction='minimize')
        study.optimize(lambda t: objective(t, X_train, y_train, X_test, y_test), n_trials=20)

        print("--- 3. Обучение финальной модели ---")
        # Снова включаем автологирование для финального обучения
        mlflow.xgboost.autolog(disable=False, log_models=False)
        
        # Логируем лучшие параметры в основной run
        mlflow.log_params(study.best_params)
        
        final_model = xgb.XGBRegressor(**study.best_params, random_state=42, tree_method='hist')
        final_model.fit(X_train, y_train)
        
        y_pred = final_model.predict(X_test)
        metrics = {
            "mae": mean_absolute_error(y_test, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
            "r2": r2_score(y_test, y_pred)
        }
        mlflow.log_metrics(metrics)
        print(f"Финальные результаты: MAE={metrics['mae']:.2f}, R2={metrics['r2']:.2f}")

        # --- Регистрация модели ---
        signature = infer_signature(X_test, y_pred)
        mlflow.xgboost.log_model(
            final_model, 
            artifact_path="model",
            signature=signature,
            registered_model_name="Energy_Forecast_XGB"
        )
        
        print(f"Модель зарегистрирована. Run ID: {run.info.run_id}")

    print("\nОбучение успешно завершено.")

if __name__ == "__main__":
    main()