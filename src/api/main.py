from time import perf_counter

from fastapi import FastAPI, Request

from src.api.metrics import HTTP_EXCEPTIONS, HTTP_IN_PROGRESS, HTTP_REQUEST_DURATION, HTTP_REQUESTS
from src.api.routes import admin, health, model, retrain

app = FastAPI(title="Energy Forecast API")


def _path_template(request: Request) -> str:
    route = request.scope.get("route")
    if route is not None and hasattr(route, "path"):
        return route.path
    return request.url.path


@app.middleware("http")
async def collect_http_metrics(request: Request, call_next):
    method = request.method
    start = perf_counter()
    HTTP_IN_PROGRESS.inc()
    try:
        response = await call_next(request)
    except Exception as exc:
        path = _path_template(request)
        HTTP_EXCEPTIONS.labels(method=method, path=path, exception_type=exc.__class__.__name__).inc()
        HTTP_REQUESTS.labels(method=method, path=path, status="500").inc()
        HTTP_REQUEST_DURATION.labels(method=method, path=path).observe(perf_counter() - start)
        HTTP_IN_PROGRESS.dec()
        raise

    path = _path_template(request)
    HTTP_REQUESTS.labels(method=method, path=path, status=str(response.status_code)).inc()
    HTTP_REQUEST_DURATION.labels(method=method, path=path).observe(perf_counter() - start)
    HTTP_IN_PROGRESS.dec()
    return response


app.include_router(health.router)
app.include_router(model.router)
app.include_router(retrain.router)
app.include_router(admin.router)
