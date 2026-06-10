try:
    from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
except ImportError:  # pragma: no cover
    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"

    class _NoopMetric:
        def inc(self, *args, **kwargs):
            return None

        def set(self, value):
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
