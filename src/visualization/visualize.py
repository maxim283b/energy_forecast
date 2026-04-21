import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.xgboost
from pathlib import Path
import numpy as np

# Настройка путей
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT_DIR / 'data/processed/energy_ready.csv'
REPORT_DIR = ROOT_DIR / 'reports/figures'
TRACKING_URI = "http://127.0.0.1:5000"
MODEL_NAME = "Energy_Forecast_XGB"

def plot_feature_importance(model, feature_names):
    """
    Визуализация важности признаков. 
    Работает с нативным XGBRegressor загруженным через mlflow.xgboost
    """
    # Для XGBRegressor важность всегда в .feature_importances_
    # Если это Booster (редко), берем через get_score()
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        # Фолбэк для нативного бустера
        score = model.get_booster().get_score(importance_type='gain')
        importances = [score.get(f, 0) for f in feature_names]
    
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=importance_df.head(15), palette='viridis')
    plt.title('Top 15 Feature Importance (Gain)')
    plt.tight_layout()
    
    file_path = REPORT_DIR / 'feature_importance.png'
    plt.savefig(file_path)
    plt.close()
    
    mlflow.log_artifact(str(file_path))
    print(f"-> График важности сохранен: {file_path}")

def plot_predictions(y_true, y_pred):
    """График сравнения прогноза и факта"""
    plt.figure(figsize=(14, 6))
    
    actual = y_true.values[-168:]
    predicted = y_pred[-168:]
    
    plt.plot(actual, label='Actual Price', color='royalblue', linewidth=2, alpha=0.8)
    plt.plot(predicted, label='Predicted Price', color='darkorange', linestyle='--', linewidth=2)
    
    plt.title('Energy Price: Actual vs Predicted (Last 7 Days)')
    plt.xlabel('Time (Hours)')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    file_path = REPORT_DIR / 'actual_vs_pred.png'
    plt.savefig(file_path)
    plt.close()
    
    mlflow.log_artifact(str(file_path))
    print(f"-> График прогнозов сохранен: {file_path}")

def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(TRACKING_URI)
    
    print("--- 1. Загрузка модели и данных ---")
    model_uri = f"models:/{MODEL_NAME}/latest"
    
    # КРИТИЧЕСКОЕ ИЗМЕНЕНИЕ: Загружаем именно через mlflow.xgboost
    # Это вернет нам объект XGBRegressor со всеми его атрибутами
    try:
        model = mlflow.xgboost.load_model(model_uri)
    except Exception as e:
        print(f"Не удалось загрузить как xgboost, пробуем pyfunc: {e}")
        model = mlflow.pyfunc.load_model(model_uri)

    # Достаем признаки
    if hasattr(model, "feature_names_in_"):
        # Если модель от Scikit-Learn API
        expected_features = model.feature_names_in_.tolist()
    else:
        # Пытаемся достать из бустера
        expected_features = model.get_booster().feature_names
        
    print(f"Модель ожидает {len(expected_features)} признаков")

    df = pd.read_csv(DATA_PATH)
    X = df[expected_features].astype(np.float64)
    y = df['price']
    
    print("--- 2. Генерация отчета ---")
    preds = model.predict(X)
    
    with mlflow.start_run(run_name="Visualization_Report"):
        plot_feature_importance(model, expected_features)
        plot_predictions(y, preds)

    print("\nВизуализация успешно завершена.")

if __name__ == "__main__":
    main()