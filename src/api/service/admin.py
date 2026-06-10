import re
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import pandas as pd
from fastapi import UploadFile

from src.api import settings
from src.api.metrics import (
    DATASET_LAST_UPLOAD_FILE_SIZE_BYTES,
    DATASET_LAST_UPLOAD_ROWS,
    DATASET_LAST_UPLOAD_TIMESTAMP,
    DATASET_UPLOAD_DURATION,
    DATASET_UPLOAD_FILE_SIZE_BYTES,
    DATASET_UPLOAD_REQUESTS,
    DATASET_UPLOAD_ROWS,
)
from src.api.schemas import EntsoeFetchRequest
from src.data.data_loader import EnergyDataGoldMiner
from src.data.make_dataset import clean_dataset
from src.features.build_features import build_feature_dataset
from src.models import predict_model
from src.monitoring import generate_reports

REQUIRED_UPLOAD_COLUMNS = {
    "timestamp",
    "price",
    "load_forecast",
    "solar_forecast",
    "wind_forecast",
    "temperature_2m",
    "wind_speed_10m",
    "direct_radiation",
}

ADMIN_JOBS: dict[str, dict] = {}


class AdminServiceError(Exception):
    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(detail)


def _sanitize_filename(filename: str) -> str:
    safe_name = Path(filename or "dataset.csv").name
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", safe_name)
    return safe_name or "dataset.csv"


def _validate_uploaded_csv(csv_path: Path) -> None:
    header = pd.read_csv(csv_path, nrows=1)
    missing_columns = sorted(REQUIRED_UPLOAD_COLUMNS.difference(header.columns))
    if missing_columns:
        raise AdminServiceError(400, f"Dataset is missing required columns: {', '.join(missing_columns)}")


def _new_job(job_type: str, message: str) -> dict:
    job_id = uuid.uuid4().hex
    job = {
        "status": "running",
        "job_id": job_id,
        "job_type": job_type,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "finished_at": None,
        "progress": 5,
        "stage": "queued",
        "message": message,
        "result": None,
    }
    ADMIN_JOBS[job_id] = job
    return job


def _update_job(job: dict, progress: int, stage: str, message: str) -> None:
    job["progress"] = progress
    job["stage"] = stage
    job["message"] = message


def _complete_job(job: dict, status: str, result: dict | None, message: str) -> None:
    job["status"] = status
    job["finished_at"] = datetime.now(timezone.utc).isoformat()
    job["progress"] = 100
    job["stage"] = "completed" if status == "succeeded" else "failed"
    job["message"] = message
    job["result"] = result


def _refresh_artifacts() -> tuple[bool, bool]:
    predictions_generated = False
    reports_generated = False

    try:
        predict_model.main()
        predictions_generated = settings.PREDICTIONS_PATH.exists()
    except Exception:
        predictions_generated = False

    try:
        generate_reports.main()
        reports_generated = settings.REPORTS_DIR.exists()
    except Exception:
        reports_generated = False

    return predictions_generated, reports_generated


def _process_uploaded_dataset(job: dict, filename: str, file_bytes: bytes) -> None:
    started_at = perf_counter()
    settings.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    upload_name = f"{uuid.uuid4().hex}_{_sanitize_filename(filename)}"
    raw_path = settings.RAW_DATA_DIR / upload_name

    try:
        _update_job(job, 15, "saving_raw", "Saving uploaded raw dataset.")
        raw_path.write_bytes(file_bytes)
        DATASET_UPLOAD_FILE_SIZE_BYTES.observe(len(file_bytes))

        _update_job(job, 30, "validating", "Validating uploaded dataset schema.")
        _validate_uploaded_csv(raw_path)

        _update_job(job, 45, "cleaning", "Cleaning raw dataset.")
        interim_path = clean_dataset(raw_path, settings.INTERIM_DATA_PATH)

        _update_job(job, 60, "featurizing", "Building feature dataset.")
        processed_path = build_feature_dataset(interim_path, settings.PROCESSED_DATA_PATH)
        processed_df = pd.read_csv(processed_path)

        _update_job(job, 80, "predicting", "Generating latest forecast.")
        predictions_generated, reports_generated = _refresh_artifacts()

        DATASET_UPLOAD_REQUESTS.labels(result="success").inc()
        DATASET_UPLOAD_DURATION.observe(perf_counter() - started_at)
        DATASET_UPLOAD_ROWS.observe(len(processed_df))
        DATASET_LAST_UPLOAD_ROWS.set(len(processed_df))
        DATASET_LAST_UPLOAD_FILE_SIZE_BYTES.set(len(file_bytes))
        DATASET_LAST_UPLOAD_TIMESTAMP.set(datetime.now(timezone.utc).timestamp())
        result = {
            "status": "uploaded",
            "filename": filename,
            "raw_path": str(raw_path.relative_to(settings.BASE_DIR)),
            "interim_path": str(Path(interim_path).relative_to(settings.BASE_DIR)),
            "processed_path": str(Path(processed_path).relative_to(settings.BASE_DIR)),
            "processed_rows": int(len(processed_df)),
            "ready_for_retrain": True,
            "predictions_generated": predictions_generated,
            "reports_generated": reports_generated,
        }
        _complete_job(job, "succeeded", result, "Uploaded dataset processed successfully.")
    except AdminServiceError as exc:
        DATASET_UPLOAD_REQUESTS.labels(result="rejected").inc()
        DATASET_UPLOAD_DURATION.observe(perf_counter() - started_at)
        _complete_job(job, "failed", None, exc.detail)
    except Exception as exc:
        DATASET_UPLOAD_REQUESTS.labels(result="error").inc()
        DATASET_UPLOAD_DURATION.observe(perf_counter() - started_at)
        _complete_job(job, "failed", None, f"Failed to process uploaded dataset: {exc}")


def _process_entsoe_fetch(job: dict, request: EntsoeFetchRequest) -> None:
    started_at = perf_counter()
    settings.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    raw_name = f"{uuid.uuid4().hex}_entsoe_{request.country_code.lower()}_{request.start_year}_{request.end_year}.csv"
    raw_path = settings.RAW_DATA_DIR / raw_name

    try:
        if request.start_year > request.end_year:
            raise AdminServiceError(400, "start_year must be less than or equal to end_year")

        _update_job(job, 10, "connecting_entsoe", "Connecting to ENTSO-E API.")
        miner = EnergyDataGoldMiner()
        _update_job(job, 35, "fetching_entsoe", "Fetching current data from ENTSO-E.")
        dataset = miner.fetch_year_range_data(
            request.country_code,
            request.lat,
            request.lon,
            request.start_year,
            request.end_year,
        )
        raw_path.write_bytes(dataset.to_csv(index=False).encode("utf-8"))

        _update_job(job, 45, "validating", "Validating fetched dataset.")
        _validate_uploaded_csv(raw_path)
        _update_job(job, 60, "cleaning", "Cleaning fetched dataset.")
        interim_path = clean_dataset(raw_path, settings.INTERIM_DATA_PATH)
        _update_job(job, 75, "featurizing", "Building features from fetched dataset.")
        processed_path = build_feature_dataset(interim_path, settings.PROCESSED_DATA_PATH)
        processed_df = pd.read_csv(processed_path)
        _update_job(job, 90, "predicting", "Generating predictions and drift reports.")
        predictions_generated, reports_generated = _refresh_artifacts()

        DATASET_UPLOAD_REQUESTS.labels(result="success").inc()
        DATASET_UPLOAD_DURATION.observe(perf_counter() - started_at)
        DATASET_UPLOAD_ROWS.observe(len(processed_df))
        DATASET_LAST_UPLOAD_ROWS.set(len(processed_df))
        DATASET_LAST_UPLOAD_FILE_SIZE_BYTES.set(raw_path.stat().st_size)
        DATASET_LAST_UPLOAD_TIMESTAMP.set(datetime.now(timezone.utc).timestamp())
        result = {
            "status": "fetched",
            "country_code": request.country_code,
            "start_year": request.start_year,
            "end_year": request.end_year,
            "raw_path": str(raw_path.relative_to(settings.BASE_DIR)),
            "interim_path": str(Path(interim_path).relative_to(settings.BASE_DIR)),
            "processed_path": str(Path(processed_path).relative_to(settings.BASE_DIR)),
            "processed_rows": int(len(processed_df)),
            "ready_for_retrain": True,
            "predictions_generated": predictions_generated,
            "reports_generated": reports_generated,
        }
        _complete_job(job, "succeeded", result, "ENTSO-E data fetched and processed successfully.")
    except AdminServiceError as exc:
        DATASET_UPLOAD_REQUESTS.labels(result="rejected").inc()
        DATASET_UPLOAD_DURATION.observe(perf_counter() - started_at)
        _complete_job(job, "failed", None, exc.detail)
    except Exception as exc:
        DATASET_UPLOAD_REQUESTS.labels(result="error").inc()
        DATASET_UPLOAD_DURATION.observe(perf_counter() - started_at)
        _complete_job(job, "failed", None, f"Failed to fetch ENTSO-E dataset: {exc}")


def _process_artifacts(job: dict) -> None:
    try:
        _update_job(job, 20, "predicting", "Generating latest forecast.")
        predictions_generated, reports_generated = _refresh_artifacts()
        result = {
            "status": "generated",
            "predictions_generated": predictions_generated,
            "reports_generated": reports_generated,
            "predictions_path": str(settings.PREDICTIONS_PATH.relative_to(settings.BASE_DIR))
            if settings.PREDICTIONS_PATH.exists()
            else None,
            "reports_dir": str(settings.REPORTS_DIR.relative_to(settings.BASE_DIR))
            if settings.REPORTS_DIR.exists()
            else None,
        }
        _complete_job(job, "succeeded", result, "Predictions and reports generated successfully.")
    except Exception as exc:
        _complete_job(job, "failed", None, f"Artifacts generation failed: {exc}")


def upload_dataset(file: UploadFile) -> dict:
    filename = file.filename or ""
    if not filename.lower().endswith(".csv"):
        raise AdminServiceError(400, "Only CSV uploads are supported")
    try:
        file_bytes = file.file.read()
    finally:
        file.file.close()

    job = _new_job("dataset_upload", "Dataset upload job created.")
    threading.Thread(target=_process_uploaded_dataset, args=(job, filename, file_bytes), daemon=True).start()
    return {
        "status": "started",
        "job_id": job["job_id"],
        "job_type": job["job_type"],
        "status_url": f"/v1/admin/jobs/{job['job_id']}",
    }


def fetch_entsoe_dataset(request: EntsoeFetchRequest) -> dict:
    job = _new_job("entsoe_fetch", "ENTSO-E fetch job created.")
    threading.Thread(target=_process_entsoe_fetch, args=(job, request), daemon=True).start()
    return {
        "status": "started",
        "job_id": job["job_id"],
        "job_type": job["job_type"],
        "status_url": f"/v1/admin/jobs/{job['job_id']}",
    }


def generate_artifacts() -> dict:
    job = _new_job("artifacts_generate", "Artifacts generation job created.")
    threading.Thread(target=_process_artifacts, args=(job,), daemon=True).start()
    return {
        "status": "started",
        "job_id": job["job_id"],
        "job_type": job["job_type"],
        "status_url": f"/v1/admin/jobs/{job['job_id']}",
    }


def get_admin_job_status(job_id: str) -> dict:
    job = ADMIN_JOBS.get(job_id)
    if job is None:
        raise AdminServiceError(404, "Admin job not found")
    return job
