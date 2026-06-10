from fastapi import APIRouter, HTTPException

from src.api.schemas import RetrainRequest, RetrainResponse, RetrainStatus
from src.api.service import retrain as retrain_service

router = APIRouter()


@router.post("/v1/retrain", response_model=RetrainResponse, status_code=202)
def retrain(request: RetrainRequest):
    try:
        return retrain_service.start_retrain(request)
    except retrain_service.RetrainServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc


@router.get("/v1/retrain/{job_id}", response_model=RetrainStatus)
def retrain_status(job_id: str):
    try:
        return retrain_service.get_retrain_status(job_id)
    except retrain_service.RetrainServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
