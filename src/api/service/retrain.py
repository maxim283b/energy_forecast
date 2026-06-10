import os
import subprocess
import sys
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from src.api import settings
from src.api.schemas import RetrainRequest, RetrainResponse, RetrainStatus
from src.api.metrics import (
    RETRAIN_ACTIVE_JOBS,
    RETRAIN_COMPLETIONS,
    RETRAIN_DURATION,
    RETRAIN_LAST_FINISHED_TIMESTAMP,
    RETRAIN_REQUESTS,
)
from src.api.service import model as model_service

RETRAIN_JOBS: dict[str, dict] = {}


class RetrainServiceError(Exception):
    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(detail)


def resolve_dataset_path(dataset: str) -> Path:
    path = Path(dataset)
    if not path.is_absolute():
        path = settings.BASE_DIR / path
    return path.resolve()


def _watch_retrain_job(job_id: str, process: subprocess.Popen, log_file, started_at_monotonic: float) -> None:
    return_code = process.wait()
    log_file.close()

    job = RETRAIN_JOBS[job_id]
    job["return_code"] = return_code
    job["finished_at"] = datetime.now(timezone.utc).isoformat()
    RETRAIN_ACTIVE_JOBS.dec()
    RETRAIN_DURATION.observe(perf_counter() - started_at_monotonic)
    RETRAIN_LAST_FINISHED_TIMESTAMP.set(datetime.now(timezone.utc).timestamp())

    if return_code == 0:
        job["model_reloaded"] = model_service.reload_model()
        job["status"] = "succeeded" if job["model_reloaded"] else "reload_failed"
    else:
        job["status"] = "failed"
    RETRAIN_COMPLETIONS.labels(status=job["status"]).inc()


def start_retrain(request: RetrainRequest) -> RetrainResponse:
    if not settings.TRAIN_SCRIPT.exists():
        RETRAIN_REQUESTS.labels(result="rejected").inc()
        raise RetrainServiceError(404, "Training script not found")

    dataset_path = resolve_dataset_path(request.dataset)
    try:
        dataset_path.relative_to(settings.BASE_DIR)
    except ValueError as exc:
        RETRAIN_REQUESTS.labels(result="rejected").inc()
        raise RetrainServiceError(400, "Dataset must be inside project") from exc

    if not dataset_path.exists():
        RETRAIN_REQUESTS.labels(result="rejected").inc()
        raise RetrainServiceError(404, "Dataset not found")

    if settings.MODEL_PATH.exists() and not request.force:
        RETRAIN_REQUESTS.labels(result="rejected").inc()
        raise RetrainServiceError(409, "Model already exists. Set force=true to retrain.")

    job_id = uuid.uuid4().hex
    settings.RETRAIN_LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = settings.RETRAIN_LOG_DIR / f"{job_id}.log"
    command = [sys.executable, str(settings.TRAIN_SCRIPT), "--dataset", str(dataset_path)]
    if request.force:
        command.append("--force")

    env = os.environ.copy()
    env["MLFLOW_TRACKING_URI"] = settings.DEFAULT_MLFLOW_TRACKING_URI
    env["RETRAIN_DATASET"] = str(dataset_path)
    env.setdefault("MLFLOW_REGISTERED_MODEL_NAME", settings.DEFAULT_REGISTERED_MODEL_NAME)

    try:
        log_file = log_path.open("w", encoding="utf-8")
        started_at_monotonic = perf_counter()
        process = subprocess.Popen(
            command,
            cwd=str(settings.BASE_DIR),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
    except Exception as exc:
        RETRAIN_REQUESTS.labels(result="error").inc()
        raise RetrainServiceError(500, f"Failed to start retrain: {exc}") from exc

    RETRAIN_REQUESTS.labels(result="accepted").inc()
    RETRAIN_ACTIVE_JOBS.inc()
    RETRAIN_JOBS[job_id] = {
        "status": "running",
        "job_id": job_id,
        "dataset": str(dataset_path.relative_to(settings.BASE_DIR)),
        "force": request.force,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "finished_at": None,
        "return_code": None,
        "log_path": str(log_path.relative_to(settings.BASE_DIR)),
        "model_reloaded": False,
    }
    if hasattr(process, "wait"):
        threading.Thread(
            target=_watch_retrain_job,
            args=(job_id, process, log_file, started_at_monotonic),
            daemon=True,
        ).start()
    else:  # pragma: no cover - used by lightweight test doubles
        log_file.close()

    return RetrainResponse(
        status="started",
        job_id=job_id,
        command=" ".join(command),
        tracking_uri=settings.DEFAULT_MLFLOW_TRACKING_URI,
        dataset=str(dataset_path.relative_to(settings.BASE_DIR)),
        force=request.force,
    )


def get_retrain_status(job_id: str) -> RetrainStatus:
    job = RETRAIN_JOBS.get(job_id)
    if job is None:
        raise RetrainServiceError(404, "Retrain job not found")
    return RetrainStatus(**job)
