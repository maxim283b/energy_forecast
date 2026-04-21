# -*- coding: utf-8 -*-
import sys
import pandas as pd
import xgboost as xgb
import mlflow
import optuna
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

# Настройка путей
BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))

# Импорт твоей функции визуализации
try:
    from src.visualization.visualize import run_visualizations
except ImportError:
    print("Warning: Не удалось импортировать run_visualizations. Проверь структуру папок.")

# Конфигурация
DATA_PATH = BASE_DIR / 'data/processed/energy_ready.csv'
MODEL_SAVE_PATH = BASE_DIR / "data/models/model.json"
REPORTS_DIR = BASE_DIR / "reports/figures"
OFFSET = 50  # Смещение для работы с отрицательными ценами

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Energy_Forecast_Final")

def objective(trial, X, y):
    """Функция оптимизации для Optuna"""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 1000, 3000),
        "max_depth": trial.suggest_int("max_depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.03, log=True),
        "subsample": trial.suggest_float("subsample", 0.7, 0.9),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 0.9),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 15),
        "gamma": trial.suggest_float("gamma", 1e-3, 2.0, log=True),
        "n_jobs": -1,
        "random_state": 42,
        "tree_method": "hist" 
    }

    tscv = TimeSeriesSplit(n_splits=5)
    scores = []

    for train_idx, val_idx in tscv.split(X):
        X_t, X_v = X.iloc[train_idx], X.iloc[val_idx]
        y_t, y_v = y.iloc[train_idx], y.iloc[val_idx]

        # Обучаем на логарифмах, но валидируем на реальных евро
        y_t_log = np.log1p(y_t + OFFSET)
        y_v_log = np.log1p(y_v + OFFSET)

        model = xgb.XGBRegressor(**params, early_stopping_rounds=50)
        model.fit(X_t, y_t_log, eval_set=[(X_v, y_v_log)], verbose=False)
        
        preds_log = model.predict(X_v)
        preds_final = np.expm1(preds_log) - OFFSET
        
        # Оптимизируем RMSE в реальных единицах (Евро)
        scores.append(np.sqrt(mean_squared_error(y_v, preds_final)))

    return np.mean(scores)

def main():
    # 1. Загрузка и подготовка
    if not DATA_PATH.exists():
        print(f"Ошибка: Файл {DATA_PATH} не найден!")
        return

    df = pd.read_csv(DATA_PATH).sort_values('timestamp')
    X = df.drop(columns=['timestamp', 'target'])
    if 'price' in X.columns: 
        X = X.drop(columns=['price'])
    y = df['target']

    # 2. Поиск гиперпараметров
    print(f"--- Запуск Optuna: Поиск лучших параметров для логарифма ---")
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, X, y), n_trials=30)

    # 3. Финальный сплит
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # Логарифмируем таргеты
    y_train_log = np.log1p(y_train + OFFSET)
    y_test_log = np.log1p(y_test + OFFSET)

    # 4. Финальное обучение и логгирование
    with mlflow.start_run(run_name="XGB_Final_Log_Corrected"):
        best_model = xgb.XGBRegressor(**study.best_params, early_stopping_rounds=100)
        
        best_model.fit(
            X_train, y_train_log, 
            eval_set=[(X_test, y_test_log)], 
            verbose=100
        )

        # 5. Обратная трансформация (ГЛАВНОЕ ДЛЯ ГРАФИКОВ)
        preds_log = best_model.predict(X_test)
        y_pred_final = np.expm1(preds_log) - OFFSET

        # Считаем метрики на РЕАЛЬНЫХ ценах
        mae = mean_absolute_error(y_test, y_pred_final)
        r2 = r2_score(y_test, y_pred_final)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred_final))
        
        # Логируем в MLflow
        mlflow.log_params(study.best_params)
        mlflow.log_metrics({"mae": mae, "r2": r2, "rmse": rmse})
        mlflow.xgboost.log_model(best_model, "model")

        # 6. Сохранение и Визуализация
        MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
        best_model.save_model(str(MODEL_SAVE_PATH))

        # Подготовка данных для визуализации (только реальные евро!)
        test_df_viz = df.iloc[split_idx:].copy()
        test_df_viz['prediction'] = y_pred_final 
        
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        run_visualizations(best_model, test_df_viz, REPORTS_DIR)
        
        print(f"\n--- Финальные результаты ---")
        print(f"MAE:  {mae:.4f}")
        print(f"R2:   {r2:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"Модель сохранена: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()