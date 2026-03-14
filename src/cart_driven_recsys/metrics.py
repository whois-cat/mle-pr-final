from __future__ import annotations
import time
from prometheus_client import Counter, Histogram


HTTP_REQUESTS_TOTAL = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["path", "method", "status"],
)

HTTP_REQUEST_DURATION_SECONDS = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["path", "method"],
)

RECSYS_RECOMMENDATIONS_TOTAL = Counter(
    "recsys_recommendations_total",
    "Total recommendation requests served",
)

RECSYS_RECOMMENDATION_ERRORS_TOTAL = Counter(
    "recsys_recommendation_errors_total",
    "Total failed recommendation requests",
)

RECSYS_RECOMMENDATION_DURATION_SECONDS = Histogram(
    "recsys_recommendation_duration_seconds",
    "Time to generate recommendations",
)

TRAINING_RUNS_TOTAL = Counter(
    "recsys_training_runs_total",
    "Total training runs",
    ["status"],
)

TRAINING_DURATION_SECONDS = Histogram(
    "recsys_training_duration_seconds",
    "Total training duration in seconds",
)


class RequestTimer:
    def __init__(self, *, path: str, method: str):
        self.path = path
        self.method = method
        self._start = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        duration = time.perf_counter() - self._start
        HTTP_REQUEST_DURATION_SECONDS.labels(path=self.path, method=self.method).observe(duration)
