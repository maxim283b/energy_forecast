from __future__ import annotations

import json
import os
import sys
import time
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
    ADMIN_DATASET_UPLOAD_URL,
    ADMIN_ENTSOE_FETCH_URL,
    ADMIN_GENERATE_ARTIFACTS_URL,
    ADMIN_JOB_STATUS_API_BASE,
    MLFLOW_EXPERIMENT,
    MLFLOW_TRACKING_URI,
    PREDICTIONS_PATH,
    REPORTS_DIR,
    RETRAIN_API_URL,
    RETRAIN_DATASET,
    RETRAIN_STATUS_API_BASE,
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
        predictions["anomaly_flag"] = (predictions["predicted_price"] < 0) | (predictions["predicted_price"] > 200)

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
        job_id = data.get("job_id")
        st.success(f"Retrain started. job_id: {job_id or 'n/a'}")
        if job_id:
            _track_job_progress(
                f"{RETRAIN_STATUS_API_BASE}/{job_id}",
                title="Retraining Progress",
                completion_message="Retraining job finished.",
            )
        st.json(data)
    except requests.RequestException as exc:
        st.error(f"Retrain API error: {exc}")
        st.info("Run the API in another terminal: " "`uvicorn src.api.main:app --host 127.0.0.1 --port 8001`")


def _track_job_progress(status_url: str, title: str, completion_message: str) -> None:
    progress_placeholder = st.empty()
    status_placeholder = st.empty()
    details_placeholder = st.empty()

    progress_bar = progress_placeholder.progress(0, text=title)
    for _ in range(240):
        try:
            response = requests.get(status_url, timeout=10)
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as exc:
            status_placeholder.error(f"Status polling failed: {exc}")
            return

        progress = int(data.get("progress", 0))
        stage = data.get("stage", "running")
        message = data.get("message") or stage
        status = data.get("status", "running")

        progress_bar.progress(max(0, min(progress, 100)), text=f"{title}: {message}")
        status_placeholder.caption(f"Stage: `{stage}` | Status: `{status}`")
        details_placeholder.json(data)

        if status in {"succeeded", "failed", "reload_failed"}:
            if status == "succeeded":
                st.success(completion_message)
            else:
                st.error(f"{title} ended with status: {status}")
            return

        time.sleep(2)

    status_placeholder.warning(f"{title} is still running. Refresh the page to continue tracking.")


def render_admin_upload() -> None:
    st.subheader("Admin Dataset Upload")
    st.caption(f"POST {ADMIN_DATASET_UPLOAD_URL}")
    uploaded_file = st.file_uploader("Upload raw dataset (.csv)", type=["csv"], key="admin_dataset_upload")
    if uploaded_file is None:
        st.info("Upload a CSV to refresh raw/interim/processed datasets.")
        return

    if st.button("Process Uploaded Dataset"):
        try:
            uploaded_file.seek(0)
            response = requests.post(
                ADMIN_DATASET_UPLOAD_URL,
                files={"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")},
                timeout=120,
            )
            response.raise_for_status()
            data = response.json()
            st.success("Dataset upload job started.")
            job_id = data.get("job_id")
            if job_id:
                _track_job_progress(
                    f"{ADMIN_JOB_STATUS_API_BASE}/{job_id}",
                    title="Dataset Upload Progress",
                    completion_message="Dataset upload and processing finished.",
                )
            st.json(data)
        except requests.RequestException as exc:
            st.error(f"Upload API error: {exc}")
            st.info("Run the API in another terminal: " "`uvicorn src.api.main:app --host 127.0.0.1 --port 8001`")


def render_admin_entsoe_fetch() -> None:
    st.subheader("Fetch Current Data From ENTSO-E")
    st.caption(f"POST {ADMIN_ENTSOE_FETCH_URL}")

    current_year = pd.Timestamp.utcnow().year
    with st.form("entsoe_fetch_form"):
        country_code = st.text_input("Country code", value="BE")
        lat = st.number_input("Latitude", value=50.85, format="%.4f")
        lon = st.number_input("Longitude", value=4.35, format="%.4f")
        start_year = st.number_input("Start year", min_value=2015, max_value=current_year, value=current_year - 1)
        end_year = st.number_input("End year", min_value=2015, max_value=current_year, value=current_year)
        submitted = st.form_submit_button("Fetch ENTSO-E Dataset")

    if not submitted:
        st.info("Fetch fresh raw data from ENTSO-E, then rebuild processed data, predictions, and drift reports.")
        return

    payload = {
        "country_code": country_code.strip().upper(),
        "lat": lat,
        "lon": lon,
        "start_year": int(start_year),
        "end_year": int(end_year),
    }
    try:
        response = requests.post(ADMIN_ENTSOE_FETCH_URL, json=payload, timeout=600)
        response.raise_for_status()
        data = response.json()
        st.success("ENTSO-E fetch job started.")
        job_id = data.get("job_id")
        if job_id:
            _track_job_progress(
                f"{ADMIN_JOB_STATUS_API_BASE}/{job_id}",
                title="ENTSO-E Fetch Progress",
                completion_message="ENTSO-E data fetch and processing finished.",
            )
        st.json(data)
    except requests.RequestException as exc:
        st.error(f"ENTSO-E fetch error: {exc}")


def render_admin_artifacts() -> None:
    st.subheader("Generate Predictions And Reports")
    st.caption(f"POST {ADMIN_GENERATE_ARTIFACTS_URL}")
    if st.button("Generate Latest Forecast And Drift Reports"):
        try:
            response = requests.post(ADMIN_GENERATE_ARTIFACTS_URL, timeout=300)
            response.raise_for_status()
            data = response.json()
            st.success("Artifacts generation job started.")
            job_id = data.get("job_id")
            if job_id:
                _track_job_progress(
                    f"{ADMIN_JOB_STATUS_API_BASE}/{job_id}",
                    title="Artifacts Generation Progress",
                    completion_message="Predictions and drift reports generated.",
                )
            st.json(data)
        except requests.RequestException as exc:
            st.error(f"Artifacts generation error: {exc}")


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

    tab_predictions, tab_drift, tab_quality, tab_experiments, tab_admin = st.tabs(
        [
            "Inference & History",
            "Data & Target Drift",
            "Model Quality",
            "Experiments",
            "Admin",
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

    with tab_admin:
        render_admin_entsoe_fetch()
        render_admin_upload()
        render_admin_artifacts()


if __name__ == "__main__":
    if get_script_run_ctx() is None:
        os.execv(
            sys.executable,
            [sys.executable, "-m", "streamlit", "run", str(Path(__file__).resolve())],
        )
    main()
