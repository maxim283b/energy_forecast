"""
Local Retrain API (same contract as Maxim's service).
Run: uvicorn src.api.retrain_server:app --host 127.0.0.1 --port 8001
"""
from __future__ import annotations

import os
import subprocess
import sys
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parents[2]
TRAIN_SCRIPT = BASE_DIR / "src/models/train_optuna.py"
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5050")

app = FastAPI(title="Energy Forecast Retrain API", version="1.0.0")
_jobs: dict[str, subprocess.Popen | None] = {}


class RetrainRequest(BaseModel):
    dataset: str = "data/processed/energy_ready.csv"
    force: bool = True


class RetrainResponse(BaseModel):
    status: str
    job_id: str
    command: str
    tracking_uri: str
    dataset: str
    force: bool


@app.post("/v1/retrain", response_model=RetrainResponse)
def start_retrain(body: RetrainRequest) -> RetrainResponse:
    dataset_path = BASE_DIR / body.dataset
    if not dataset_path.exists() and not body.force:
        raise HTTPException(status_code=400, detail=f"Dataset not found: {dataset_path}")

    job_id = str(uuid.uuid4())
    command = f"{sys.executable} {TRAIN_SCRIPT}"
    env = {**os.environ, "MLFLOW_TRACKING_URI": MLFLOW_URI}

    proc = subprocess.Popen(
        [sys.executable, str(TRAIN_SCRIPT)],
        cwd=str(BASE_DIR),
        env=env,
    )
    _jobs[job_id] = proc

    return RetrainResponse(
        status="started",
        job_id=job_id,
        command=command,
        tracking_uri=MLFLOW_URI,
        dataset=body.dataset,
        force=body.force,
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
