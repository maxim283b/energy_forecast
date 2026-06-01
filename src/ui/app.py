from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import mlflow
import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components
from streamlit.runtime.scriptrunner import get_script_run_ctx

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))

from src.monitoring.config import (  # noqa: E402
    MLFLOW_EXPERIMENT,
    MLFLOW_TRACKING_URI,
    PREDICTIONS_PATH,
    REPORTS_DIR,
    RETRAIN_API_URL,
    RETRAIN_DATASET,
    TRIGGER_STATUS_PATH,
)


def load_trigger_status() -> dict | None:
    if not TRIGGER_STATUS_PATH.exists():
        return None
    return json.loads(TRIGGER_STATUS_PATH.read_text(encoding="utf-8"))


def render_drift_notifications() -> None:
    status = load_trigger_status()
    if status is None:
        st.info("Drift status is not available yet.")
        return

    checked_at = status.get("checked_at", "n/a")
    if status.get("should_retrain"):
        st.error(f"Drift alert. Retrain is recommended. Checked at: {checked_at}")
        for reason in status.get("reasons", []):
            st.warning(reason)
    else:
        st.success(f"No active drift alert. Checked at: {checked_at}")


def render_report(report_name: str, title: str) -> None:
    report_path = REPORTS_DIR / f"{report_name}.html"
    st.subheader(title)
    if report_path.exists():
        html = report_path.read_text(encoding="utf-8")
        components.html(html, height=700, scrolling=True)
    else:
        st.info(f"Report not found: {report_path}")


def render_predictions() -> None:
    st.subheader("Latest Forecast")
    if not PREDICTIONS_PATH.exists():
        st.warning("No local predictions yet. Run local inference first.")
        return

    predictions = pd.read_csv(PREDICTIONS_PATH)
    if "predicted_price" in predictions.columns and "anomaly_flag" not in predictions:
        predictions["anomaly_flag"] = (predictions["predicted_price"] < 0) | (
            predictions["predicted_price"] > 200
        )

    st.dataframe(predictions, use_container_width=True)

    if {"timestamp", "predicted_price"}.issubset(predictions.columns):
        chart_df = predictions.copy()
        chart_df["timestamp"] = pd.to_datetime(chart_df["timestamp"], errors="coerce")
        chart_df = chart_df.dropna(subset=["timestamp"]).set_index("timestamp")
        st.line_chart(chart_df["predicted_price"])


def render_trigger_status() -> None:
    st.subheader("Retrain Trigger Status")
    status = load_trigger_status()
    if status is None:
        st.info("Trigger status not available yet. Generate drift reports first.")
        return

    if status.get("should_retrain"):
        st.error("Retrain recommended")
        for reason in status.get("reasons", []):
            st.write(f"- {reason}")
    else:
        st.success("Retrain not required")

    with st.expander("Details"):
        st.json(status)


def call_retrain_api() -> None:
    payload = {"dataset": RETRAIN_DATASET, "force": True}
    try:
        response = requests.post(RETRAIN_API_URL, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        st.success(f"Retrain started. job_id: {data.get('job_id', 'n/a')}")
        st.json(data)
    except requests.RequestException as exc:
        st.error(f"Retrain API error: {exc}")
        st.info(
            "Run the API in another terminal: "
            "`uvicorn src.api.main:app --host 127.0.0.1 --port 8001`"
        )


def render_experiments() -> None:
    st.subheader("MLflow Experiments")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    try:
        experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT)
        if experiment is None:
            st.info(f"Experiment not found: {MLFLOW_EXPERIMENT}")
            return

        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"],
            max_results=20,
        )
    except Exception as exc:
        st.error(f"MLflow is not available: {exc}")
        st.caption(f"Tracking URI: {MLFLOW_TRACKING_URI}")
        return

    st.caption(f"Tracking URI: {MLFLOW_TRACKING_URI}")
    st.caption(f"Experiment: {MLFLOW_EXPERIMENT}")

    if runs.empty:
        st.info("No runs found.")
        return

    preferred_columns = [
        "run_id",
        "status",
        "start_time",
        "metrics.mae",
        "metrics.rmse",
        "metrics.r2",
        "params.dataset",
        "params.force_retrain",
    ]
    visible_columns = [col for col in preferred_columns if col in runs.columns]
    st.dataframe(runs[visible_columns], use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="Energy Forecast UI", layout="wide")
    st.title("Energy Forecast Dashboard")
    st.caption("Inference, drift monitoring, model quality, and retraining")

    render_drift_notifications()

    tab_predictions, tab_drift, tab_quality, tab_experiments = st.tabs(
        [
            "Inference & History",
            "Data & Target Drift",
            "Model Quality",
            "Experiments",
        ]
    )

    with tab_predictions:
        render_predictions()

    with tab_drift:
        render_report("data_drift", "Data Drift (Evidently)")
        render_report("target_drift", "Target Drift (Evidently)")

    with tab_quality:
        render_report("regression_quality", "Regression Quality (Evidently)")
        render_trigger_status()

        st.subheader("Manual Retraining")
        st.caption(f"POST {RETRAIN_API_URL}")
        if st.button("Start Retraining"):
            call_retrain_api()

    with tab_experiments:
        render_experiments()


if __name__ == "__main__":
    if get_script_run_ctx() is None:
        os.execv(
            sys.executable,
            [sys.executable, "-m", "streamlit", "run", str(Path(__file__).resolve())],
        )
    main()
