import re
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
from src.data.make_dataset import clean_dataset
from src.features.build_features import build_feature_dataset

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


def upload_dataset(file: UploadFile) -> dict:
    started_at = perf_counter()
    filename = file.filename or ""
    if not filename.lower().endswith(".csv"):
        DATASET_UPLOAD_REQUESTS.labels(result="rejected").inc()
        raise AdminServiceError(400, "Only CSV uploads are supported")

    settings.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    upload_name = f"{uuid.uuid4().hex}_{_sanitize_filename(filename)}"
    raw_path = settings.RAW_DATA_DIR / upload_name

    try:
        file_bytes = file.file.read()
        raw_path.write_bytes(file_bytes)
        DATASET_UPLOAD_FILE_SIZE_BYTES.observe(len(file_bytes))
        _validate_uploaded_csv(raw_path)

        interim_path = clean_dataset(raw_path, settings.INTERIM_DATA_PATH)
        processed_path = build_feature_dataset(interim_path, settings.PROCESSED_DATA_PATH)
        processed_df = pd.read_csv(processed_path)
        DATASET_UPLOAD_REQUESTS.labels(result="success").inc()
        DATASET_UPLOAD_DURATION.observe(perf_counter() - started_at)
        DATASET_UPLOAD_ROWS.observe(len(processed_df))
        DATASET_LAST_UPLOAD_ROWS.set(len(processed_df))
        DATASET_LAST_UPLOAD_FILE_SIZE_BYTES.set(len(file_bytes))
        DATASET_LAST_UPLOAD_TIMESTAMP.set(datetime.now(timezone.utc).timestamp())
    except AdminServiceError:
        DATASET_UPLOAD_REQUESTS.labels(result="rejected").inc()
        DATASET_UPLOAD_DURATION.observe(perf_counter() - started_at)
        raise
    except Exception as exc:
        DATASET_UPLOAD_REQUESTS.labels(result="error").inc()
        DATASET_UPLOAD_DURATION.observe(perf_counter() - started_at)
        raise AdminServiceError(500, f"Failed to process uploaded dataset: {exc}") from exc
    finally:
        file.file.close()

    return {
        "status": "uploaded",
        "filename": filename,
        "raw_path": str(raw_path.relative_to(settings.BASE_DIR)),
        "interim_path": str(Path(interim_path).relative_to(settings.BASE_DIR)),
        "processed_path": str(Path(processed_path).relative_to(settings.BASE_DIR)),
        "processed_rows": int(len(processed_df)),
        "ready_for_retrain": True,
    }
