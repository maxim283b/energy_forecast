from fastapi import FastAPI

from src.api.routes import admin, health, model, retrain

app = FastAPI(title="Energy Forecast API")

app.include_router(health.router)
app.include_router(model.router)
app.include_router(retrain.router)
app.include_router(admin.router)
