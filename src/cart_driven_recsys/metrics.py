from __future__ import annotations

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