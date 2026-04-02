import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import mlflow
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.parent


def load_and_prepare_data(file_path):
    """Загрузить и подготовить данные"""
    df = pd.read_csv(file_path)

    if df.columns[0] == 'Unnamed: 0':
        df.columns = ['timestamp', 'price']
    else:
        df.columns = ['timestamp', 'price']

    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df['timestamp'] = df['timestamp'].dt.tz_convert('Europe/Brussels')

    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['day_of_month'] = df['timestamp'].dt.day
    df['month'] = df['timestamp'].dt.month
    df['weekend'] = (df['day_of_week'] >= 5).astype(int)

    return df


def setup_mlflow():
    """Настроить MLflow"""
    mlflow_dir = ROOT_DIR / 'mlflow'
    mlflow_dir.mkdir(exist_ok=True)
    mlflow.set_tracking_uri(f"file://{mlflow_dir}")

    experiment_name = "Energy Price Comparison"
    experiment = mlflow.get_experiment_by_name(experiment_name)

    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
        print(f"Created experiment: {experiment_name}")
    else:
        experiment_id = experiment.experiment_id
        print(f"Using existing experiment: {experiment_name}")

    return experiment_id


def main():
    """Основная функция сравнения моделей"""
    data_path = ROOT_DIR / 'data/raw/entsoe_prices_2024.csv'

    if not data_path.exists():
        print(f"Data file not found: {data_path}")
        print("Please run entsoe_parser.py first")
        return

    df = load_and_prepare_data(data_path)
    print(f"Data loaded: {len(df)} records")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

    feature_columns = ['hour', 'day_of_week', 'day_of_month', 'month', 'weekend']
    X = df[feature_columns]
    y = df['price']

    split_idx = int(len(df) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"Train set: {len(X_train)} records")
    print(f"Test set: {len(X_test)} records")

    experiment_id = setup_mlflow()

    param_grid = {
        'n_estimators': [50, 100, 150, 200],
        'max_depth': [5, 10, 15, 20]
    }

    results = []

    print("\nRunning experiments...")
    print("-" * 50)

    for n_estimators in param_grid['n_estimators']:
        for max_depth in param_grid['max_depth']:
            run_name = f"RF_n{n_estimators}_d{max_depth}"
            print(f"\nExperiment: {run_name}")

            with mlflow.start_run(experiment_id=experiment_id, run_name=run_name):
                model = RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=42,
                    n_jobs=-1
                )
                model.fit(X_train, y_train)

                y_pred = model.predict(X_test)
                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))

                mlflow.log_param("n_estimators", n_estimators)
                mlflow.log_param("max_depth", max_depth)
                mlflow.log_param("random_state", 42)

                mlflow.log_metric("r2_score", r2)
                mlflow.log_metric("mae", mae)
                mlflow.log_metric("rmse", rmse)

                for name, importance in zip(feature_columns, model.feature_importances_):
                    mlflow.log_metric(f"importance_{name}", importance)

                mlflow.sklearn.log_model(model, "model")

                results.append({
                    'n_estimators': n_estimators,
                    'max_depth': max_depth,
                    'r2_score': r2,
                    'mae': mae,
                    'rmse': rmse
                })

                print(f"  R2: {r2:.4f}, MAE: {mae:.2f} EUR/MWh, RMSE: {rmse:.2f} EUR/MWh")

    print("\n" + "=" * 50)
    print("BEST RESULTS:")
    print("=" * 50)

    best_results = sorted(results, key=lambda x: x['r2_score'], reverse=True)

    for i, result in enumerate(best_results[:5], 1):
        print(f"\n{i}. n_estimators={result['n_estimators']}, max_depth={result['max_depth']}")
        print(f"   R2 = {result['r2_score']:.4f}")
        print(f"   MAE = {result['mae']:.2f} EUR/MWh")
        print(f"   RMSE = {result['rmse']:.2f} EUR/MWh")

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('r2_score', ascending=False)
    results_path = ROOT_DIR / 'data/results/comparison_results.csv'
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to {results_path}")

    print("\nAll experiments completed")


if __name__ == "__main__":
    main()