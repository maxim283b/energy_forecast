import pandas as pd
import xgboost as xgb
import mlflow
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# --- Конфигурация ---
BASE_DIR = Path(__file__).resolve().parents[2]
TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "Energy_Forecast_Final_Success"
DATA_PATH = BASE_DIR / 'data/processed/energy_ready.csv'
MODEL_PATH = BASE_DIR / "data/models/model.json"

def main():
    print(f"--- Запуск расширенной визуализации ---")
    mlflow.set_tracking_uri(TRACKING_URI)
    
    # 1. Поиск Run ID и загрузка данных
    run_id = None
    exps = mlflow.search_experiments()
    target_exp = next((e for e in exps if e.name == EXPERIMENT_NAME), None)
    if target_exp:
        runs = mlflow.search_runs(experiment_ids=[target_exp.experiment_id], max_results=1)
        if not runs.empty:
            run_id = runs.iloc[0].run_id

    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=['timestamp', 'price'])
    y_true = df['price']
    
    # Загрузка модели и предсказание
    model = xgb.Booster()
    model.load_model(str(MODEL_PATH))
    dmat = xgb.DMatrix(X)
    y_pred = model.predict(dmat)

    if run_id:
        with mlflow.start_run(run_id=run_id):
            sns.set_style("whitegrid")

            # --- ГРАФИК 1: Feature Importance (уже был) ---
            importance = model.get_score(importance_type='gain')
            df_imp = pd.DataFrame({'Feature': list(importance.keys()), 'Importance': list(importance.values())}).sort_values('Importance', ascending=False).head(15)
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            sns.barplot(data=df_imp, x='Importance', y='Feature', hue='Feature', palette='magma', ax=ax1, legend=False)
            ax1.set_title("Top Features (Gain)")
            mlflow.log_figure(fig1, "plots/1_feature_importance.png")
            plt.close(fig1)

            # --- ГРАФИК 2: Predicted vs Actual (Линия идеального прогноза) ---
            fig2, ax2 = plt.subplots(figsize=(8, 8))
            ax2.scatter(y_true, y_pred, alpha=0.3, color='teal')
            ax2.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
            ax2.set_xlabel("Actual Price")
            ax2.set_ylabel("Predicted Price")
            ax2.set_title("Actual vs Predicted")
            mlflow.log_figure(fig2, "plots/2_actual_vs_predicted.png")
            plt.close(fig2)

            # --- ГРАФИК 3: Распределение ошибок (Residuals) ---
            residuals = y_true - y_pred
            fig3, ax3 = plt.subplots(figsize=(10, 6))
            sns.histplot(residuals, kde=True, color='purple', ax=ax3)
            ax3.set_title("Residuals Distribution (Errors)")
            ax3.set_xlabel("Error Value")
            mlflow.log_figure(fig3, "plots/3_residuals_hist.png")
            plt.close(fig3)

            # --- ГРАФИК 4: Ошибка во времени (Тренды) ---
            # Берем последние 100 точек для наглядности
            fig4, ax4 = plt.subplots(figsize=(14, 6))
            plt.plot(y_true.values[-100:], label='Actual', alpha=0.7)
            plt.plot(y_pred[-100:], label='Predicted', alpha=0.7, linestyle='--')
            plt.title("Last 100 Hours: Actual vs Predicted")
            plt.legend()
            mlflow.log_figure(fig4, "plots/4_time_series_zoom.png")
            plt.close(fig4)

            print(f"графики успешно загружены в MLflow Run: {run_id[:8]}")

if __name__ == "__main__":
    main()