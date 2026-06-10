from fastapi import APIRouter
from starlette.responses import Response

from src.api.metrics import CONTENT_TYPE_LATEST, generate_latest
from src.api.service import model as model_service

router = APIRouter()


@router.get("/health")
def health():
    return {"status": "ok", "model_loaded": model_service.is_model_loaded()}


@router.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
