from pydantic import BaseModel


class PredictionInput(BaseModel):
    hour_sin: float
    hour_cos: float
    day_of_week: int
    is_holiday: int
    is_weekend: int
    load_forecast: float
    net_load_forecast: float
    solar_forecast: float
    wind_forecast: float
    renewable_total: float
    non_renewable_needed: float
    load_trend_24h: float
    price_fr_lag_24: float
    price_de_lag_24: float
    price_nl_lag_24: float
    spread_be_fr_lag_24: float
    spread_be_de_lag_24: float
    spread_be_nl_lag_24: float
    temperature_2m: float
    wind_speed_10m: float
    direct_radiation: float
    price_lag_24: float
    price_lag_48: float
    price_lag_168: float
    price_mean_24h: float
    price_std_24h: float


class PredictionResponse(BaseModel):
    predicted_price: float
    anomaly_flag: bool


class RetrainRequest(BaseModel):
    dataset: str = "data/processed/energy_ready.csv"
    force: bool = False


class RetrainResponse(BaseModel):
    status: str
    job_id: str
    command: str
    tracking_uri: str
    dataset: str
    force: bool


class RetrainStatus(BaseModel):
    status: str
    job_id: str
    dataset: str
    force: bool
    started_at: str
    finished_at: str | None = None
    return_code: int | None = None
    log_path: str | None = None
    model_reloaded: bool = False


class DatasetUploadResponse(BaseModel):
    status: str
    filename: str
    raw_path: str
    interim_path: str
    processed_path: str
    processed_rows: int
    ready_for_retrain: bool


class EntsoeFetchRequest(BaseModel):
    country_code: str = "BE"
    lat: float = 50.85
    lon: float = 4.35
    start_year: int
    end_year: int


class EntsoeFetchResponse(BaseModel):
    status: str
    country_code: str
    start_year: int
    end_year: int
    raw_path: str
    interim_path: str
    processed_path: str
    processed_rows: int
    ready_for_retrain: bool
    predictions_generated: bool
    reports_generated: bool


class AdminArtifactsResponse(BaseModel):
    status: str
    predictions_generated: bool
    reports_generated: bool
    predictions_path: str | None = None
    reports_dir: str | None = None
