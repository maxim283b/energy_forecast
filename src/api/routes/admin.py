from fastapi import APIRouter, File, HTTPException, UploadFile

from src.api.schemas import EntsoeFetchRequest
from src.api.service import admin as admin_service

router = APIRouter()


@router.post("/v1/admin/dataset/upload", status_code=201)
def upload_dataset(file: UploadFile = File(...)):
    try:
        return admin_service.upload_dataset(file)
    except admin_service.AdminServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc


@router.post("/v1/admin/dataset/fetch-entsoe", status_code=201)
def fetch_entsoe_dataset(request: EntsoeFetchRequest):
    try:
        return admin_service.fetch_entsoe_dataset(request)
    except admin_service.AdminServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc


@router.post("/v1/admin/artifacts/generate", status_code=200)
def generate_artifacts():
    try:
        return admin_service.generate_artifacts()
    except admin_service.AdminServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
