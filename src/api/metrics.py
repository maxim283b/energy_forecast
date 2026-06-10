try:
    from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
except ImportError:  # pragma: no cover
    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"

    class _NoopMetric:
        def labels(self, *args, **kwargs):
            return self

        def inc(self, *args, **kwargs):
            return None

        def dec(self, *args, **kwargs):
            return None

        def set(self, value):
            return None

        def observe(self, value):
            return None

        def time(self):
            return self

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

    def Counter(*args, **kwargs):
        return _NoopMetric()

    def Gauge(*args, **kwargs):
        return _NoopMetric()

    def Histogram(*args, **kwargs):
        return _NoopMetric()

    def generate_latest():
        return b"# prometheus_client is not installed\n"


HTTP_REQUESTS = Counter(
    "energy_api_http_requests_total",
    "Total HTTP requests.",
    ["method", "path", "status"],
)
HTTP_REQUEST_DURATION = Histogram(
    "energy_api_http_request_duration_seconds",
    "HTTP request duration.",
    ["method", "path"],
)
HTTP_IN_PROGRESS = Gauge(
    "energy_api_http_in_progress_requests",
    "Current in-progress HTTP requests.",
)
HTTP_EXCEPTIONS = Counter(
    "energy_api_http_exceptions_total",
    "Unhandled HTTP exceptions.",
    ["method", "path", "exception_type"],
)

PREDICTION_REQUESTS = Counter(
    "energy_prediction_requests_total",
    "Total prediction requests.",
)
PREDICTION_ERRORS = Counter(
    "energy_prediction_errors_total",
    "Total failed prediction requests.",
)
PREDICTION_ANOMALIES = Counter(
    "energy_prediction_anomalies_total",
    "Total predictions flagged as business anomalies.",
)
PREDICTION_LATENCY = Histogram(
    "energy_prediction_latency_seconds",
    "Prediction request latency.",
)
LAST_PREDICTED_PRICE = Gauge(
    "energy_last_predicted_price",
    "Last predicted energy price.",
)
MODEL_LOADED_GAUGE = Gauge(
    "energy_model_loaded",
    "Model loaded status: 1 loaded, 0 missing or failed.",
)
MODEL_RELOADS = Counter(
    "energy_model_reload_total",
    "Total model reload attempts.",
    ["result"],
)
MODEL_RELOAD_DURATION = Histogram(
    "energy_model_reload_duration_seconds",
    "Model reload duration.",
)
MODEL_LAST_RELOAD_TIMESTAMP = Gauge(
    "energy_model_last_reload_timestamp_seconds",
    "Unix timestamp of the last successful model reload.",
)
MODEL_FILE_SIZE_BYTES = Gauge(
    "energy_model_file_size_bytes",
    "Current model file size in bytes.",
)
MODEL_FILE_MTIME_TIMESTAMP = Gauge(
    "energy_model_file_mtime_timestamp_seconds",
    "Model file modification time as Unix timestamp.",
)

RETRAIN_REQUESTS = Counter(
    "energy_retrain_requests_total",
    "Total retrain requests.",
    ["result"],
)
RETRAIN_ACTIVE_JOBS = Gauge(
    "energy_retrain_active_jobs",
    "Current number of active retrain jobs.",
)
RETRAIN_COMPLETIONS = Counter(
    "energy_retrain_completions_total",
    "Total completed retrain jobs by final status.",
    ["status"],
)
RETRAIN_DURATION = Histogram(
    "energy_retrain_duration_seconds",
    "Retrain job duration.",
)
RETRAIN_LAST_FINISHED_TIMESTAMP = Gauge(
    "energy_retrain_last_finished_timestamp_seconds",
    "Unix timestamp of the last finished retrain job.",
)

DATASET_UPLOAD_REQUESTS = Counter(
    "energy_dataset_upload_requests_total",
    "Total dataset upload requests.",
    ["result"],
)
DATASET_UPLOAD_DURATION = Histogram(
    "energy_dataset_upload_duration_seconds",
    "Dataset upload processing duration.",
)
DATASET_UPLOAD_ROWS = Histogram(
    "energy_dataset_upload_rows",
    "Processed row counts for uploaded datasets.",
)
DATASET_UPLOAD_FILE_SIZE_BYTES = Histogram(
    "energy_dataset_upload_file_size_bytes",
    "Uploaded dataset file size in bytes.",
)
DATASET_LAST_UPLOAD_ROWS = Gauge(
    "energy_dataset_last_upload_rows",
    "Row count of the most recently processed uploaded dataset.",
)
DATASET_LAST_UPLOAD_FILE_SIZE_BYTES = Gauge(
    "energy_dataset_last_upload_file_size_bytes",
    "File size in bytes of the most recently uploaded dataset.",
)
DATASET_LAST_UPLOAD_TIMESTAMP = Gauge(
    "energy_dataset_last_upload_timestamp_seconds",
    "Unix timestamp of the most recent successful dataset upload.",
)
