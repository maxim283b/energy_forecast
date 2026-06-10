from fastapi import FastAPI

from src.api.routes import health, model, retrain

app = FastAPI(title="Energy Forecast API")

app.include_router(health.router)
app.include_router(model.router)
app.include_router(retrain.router)
