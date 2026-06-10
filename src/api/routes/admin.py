from fastapi import APIRouter, File, HTTPException, UploadFile

from src.api.service import admin as admin_service

router = APIRouter()


@router.post("/v1/admin/dataset/upload", status_code=201)
def upload_dataset(file: UploadFile = File(...)):
    try:
        return admin_service.upload_dataset(file)
    except admin_service.AdminServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
