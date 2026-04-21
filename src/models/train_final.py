import mlflow
import mlflow.xgboost
import pandas as pd
import xgboost as xgb
import numpy as np
from pathlib import Path
from mlflow.models.signature import infer_signature
import os
from sklearn.metrics import (
    mean_absolute_error, 
    r2_score, 
    mean_squared_error, 
    mean_absolute_percentage_error,
    median_absolute_error,
    max_error
)

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT_DIR / 'data/processed/energy_ready.csv'

def main():
    # Настройка окружения
    os.environ["MLFLOW_ARTIFACT_PROXY_URI"] = "http://127.0.0.1:5000"
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("Energy_Forecast_Final_Success")

    # 1. Загрузка данных
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=['timestamp', 'price'])
    y = df['price']
    
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    with mlflow.start_run(run_name="Professional_Log_Full"):
        params = {
            "n_estimators": 1308,
            "max_depth": 3,
            "learning_rate": 0.014,
            "subsample": 0.72,
            "colsample_bytree": 0.84,
            "random_state": 42
        }
        mlflow.log_params(params)

        # 2. Обучение
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train)
        
        # 3. Предсказание
        y_pred = model.predict(X_test)

        # 4. Расчет расширенных метрик
        metrics = {
            "mae": mean_absolute_error(y_test, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
            "r2": r2_score(y_test, y_pred),
            "mape": mean_absolute_percentage_error(y_test, y_pred),
            "median_ae": median_absolute_error(y_test, y_pred),
            "max_error": max_error(y_test, y_pred)
        }

        # Логируем всё одним словарем
        mlflow.log_metrics(metrics)

        # 5. Логирование модели
        signature = infer_signature(X_test, y_pred)
        mlflow.xgboost.log_model(
            xgb_model=model,
            artifact_path="model",
            signature=signature,
            registered_model_name="Energy_Price_Model_V1"
        )
        
        print("-" * 30)
        for name, value in metrics.items():
            print(f"{name.upper()}: {value:.4f}")
        print("-" * 30)
        print("Все метрики и модель успешно отправлены в MLflow!")

if __name__ == "__main__":
    main()