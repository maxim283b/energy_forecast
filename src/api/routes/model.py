from fastapi import APIRouter, HTTPException

from src.api.schemas import PredictionInput, PredictionResponse
from src.api.service import model as model_service

router = APIRouter()


@router.post("/v1/model/reload")
def reload_model_endpoint():
    loaded = model_service.reload_model()
    if not loaded:
        raise HTTPException(status_code=500, detail="Model reload failed")
    return {"status": "reloaded", "model_loaded": model_service.is_model_loaded()}


@router.post("/predict", response_model=PredictionResponse)
def predict(input_data: PredictionInput):
    try:
        return model_service.predict_price(input_data)
    except model_service.ModelNotLoadedError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except model_service.ModelPredictionError as exc:
        raise HTTPException(status_code=500, detail=f"Model prediction failed: {exc}") from exc
