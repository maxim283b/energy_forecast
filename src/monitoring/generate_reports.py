from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import pandas as pd
import requests

try:
    from evidently.metric_preset import (
        DataDriftPreset,
        RegressionPreset,
        TargetDriftPreset,
    )
    from evidently.report import Report
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Evidently is not installed. Install dependencies first."
    ) from exc

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))

from src.monitoring.config import (  # noqa: E402
    BASELINE_METRICS_PATH,
    DATA_PATH,
    MLFLOW_EXPERIMENT,
    MLFLOW_TRACKING_URI,
    PREDICTIONS_PATH,
    REPORTS_DIR,
    RETRAIN_API_URL,
    RETRAIN_DATASET,
    TRIGGER_STATUS_PATH,
)
from src.monitoring.metrics import (  # noqa: E402
    add_model_predictions,
    evaluate_current_model,
    evaluate_retrain_trigger,
    save_baseline_metrics,
)


def _split_reference_current(
    df: pd.DataFrame,
    ratio: float = 0.8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_idx = int(len(df) * ratio)
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


def _save_report(
    report: Report,
    reference: pd.DataFrame,
    current: pd.DataFrame,
    name: str,
) -> None:
    report.run(reference_data=reference, current_data=current)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report.save_html(str(REPORTS_DIR / f"{name}.html"))
    report.save_json(str(REPORTS_DIR / f"{name}.json"))


def _log_reports_to_mlflow() -> bool:
    try:
        response = requests.get(f"{MLFLOW_TRACKING_URI.rstrip('/')}/health", timeout=2)
        response.raise_for_status()
    except requests.RequestException as exc:
        print(f"Warning: MLflow logging skipped ({exc})")
        return False

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    try:
        run_name = f"drift_monitoring_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}"
        with mlflow.start_run(run_name=run_name):
            for report_file in sorted(REPORTS_DIR.glob("*.html")):
                mlflow.log_artifact(str(report_file), artifact_path="drift_reports")
            for report_file in sorted(REPORTS_DIR.glob("*.json")):
                if report_file.name != "retrain_trigger.json":
                    mlflow.log_artifact(
                        str(report_file),
                        artifact_path="drift_reports",
                    )
            if TRIGGER_STATUS_PATH.exists():
                mlflow.log_artifact(
                    str(TRIGGER_STATUS_PATH),
                    artifact_path="drift_reports",
                )
    except Exception as exc:
        print(f"Warning: MLflow logging skipped ({exc})")
        return False
    return True


def _call_retrain_api() -> dict:
    payload = {"dataset": RETRAIN_DATASET, "force": True}
    response = requests.post(RETRAIN_API_URL, json=payload, timeout=30)
    response.raise_for_status()
    return response.json()


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing processed dataset: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH).sort_values("timestamp")
    if "target" not in df.columns:
        raise ValueError("Column 'target' is required for monitoring reports.")

    reference, current = _split_reference_current(df)
    feature_cols = [col for col in df.columns if col not in {"timestamp", "target"}]

    reference_data = reference[feature_cols + ["target"]].copy()
    current_data = current[feature_cols + ["target"]].copy()

    _save_report(
        Report(metrics=[DataDriftPreset()]),
        reference_data,
        current_data,
        "data_drift",
    )
    _save_report(
        Report(metrics=[TargetDriftPreset()]),
        reference_data,
        current_data,
        "target_drift",
    )

    try:
        reference_reg = add_model_predictions(reference_data)
        current_reg = add_model_predictions(current_data)
        _save_report(
            Report(metrics=[RegressionPreset()]),
            reference_reg,
            current_reg,
            "regression_quality",
        )
    except Exception as exc:
        print(f"Warning: regression_quality report skipped ({exc})")

    current_metrics = evaluate_current_model(df)
    should_set_baseline = os.getenv("SET_BASELINE", "false").lower() == "true"
    if current_metrics and (should_set_baseline or not BASELINE_METRICS_PATH.exists()):
        save_baseline_metrics(current_metrics["mae"], current_metrics["r2"])

    trigger_status = evaluate_retrain_trigger(
        reference_data,
        current_data,
        current_metrics,
    )
    trigger_status["checked_at"] = datetime.now(timezone.utc).isoformat()

    TRIGGER_STATUS_PATH.write_text(
        json.dumps(trigger_status, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(trigger_status, indent=2))

    if _log_reports_to_mlflow():
        print(f"Reports saved to {REPORTS_DIR} and logged to MLflow")
    else:
        print(f"Reports saved to {REPORTS_DIR}")

    if PREDICTIONS_PATH.exists():
        print(f"Latest predictions: {PREDICTIONS_PATH}")

    auto_retrain = os.getenv("AUTO_RETRAIN", "false").lower() == "true"
    if trigger_status["should_retrain"] and auto_retrain:
        try:
            retrain_response = _call_retrain_api()
            print(f"Auto-retrain started: {retrain_response}")
        except requests.RequestException as exc:
            print(f"Auto-retrain failed: {exc}")


if __name__ == "__main__":
    main()
