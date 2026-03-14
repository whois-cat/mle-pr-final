"""FastAPI recommendation service."""

from __future__ import annotations

import time
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Query, Request, HTTPException
from fastapi.responses import Response
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from cart_driven_recsys.config import cfg
from cart_driven_recsys.recommend import CartRecommender
from cart_driven_recsys.metrics import (
    HTTP_REQUESTS_TOTAL,
    HTTP_REQUEST_DURATION_SECONDS,
    RECSYS_RECOMMENDATIONS_TOTAL,
    RECSYS_RECOMMENDATION_ERRORS_TOTAL,
    RECSYS_RECOMMENDATION_DURATION_SECONDS,
    RequestTimer,
)

logger = logging.getLogger(__name__)
recommender: CartRecommender | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global recommender
    logger.info(f"loading model from {cfg.model_artifact}")
    recommender = CartRecommender.load(cfg.model_artifact)
    logger.info("model loaded")
    yield
    recommender = None

app = FastAPI(title="cart-recsys", version="0.1.0", lifespan=lifespan)


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    path = request.url.path
    method = request.method
    with RequestTimer(path=path, method=method):
        response = await call_next(request)
    HTTP_REQUESTS_TOTAL.labels(path=path, method=method, status=str(response.status_code)).inc()
    return response


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
def ready():
    if recommender is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return {"status": "ready"}


@app.get("/recommendations")
def recommendations(
    cart_item_ids: str = Query(default="", description="comma-separated item IDs in cart"),
    exclude_item_ids: str = Query(default="", description="comma-separated item IDs to exclude"),
    k: int = Query(default=20, ge=1, le=100),
):
    if recommender is None:
        RECSYS_RECOMMENDATION_ERRORS_TOTAL.inc()
        raise HTTPException(status_code=503, detail="model not loaded")

    cart = _parse_ids(cart_item_ids)
    exclude = set(_parse_ids(exclude_item_ids))

    t0 = time.perf_counter()
    try:
        recs = recommender.recommend(cart_item_ids=cart, n=k, exclude_item_ids=exclude)
    except Exception as exc:
        RECSYS_RECOMMENDATION_ERRORS_TOTAL.inc()
        logger.exception("recommendation failed")
        raise HTTPException(status_code=500, detail="recommendation failed") from exc

    RECSYS_RECOMMENDATION_DURATION_SECONDS.observe(time.perf_counter() - t0)
    RECSYS_RECOMMENDATIONS_TOTAL.inc()

    return {"item_ids": recs, "count": len(recs)}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


def _parse_ids(raw: str) -> list[int]:
    if not raw.strip():
        return []
    try:
        return [int(x.strip()) for x in raw.split(",") if x.strip()]
    except ValueError:
        raise HTTPException(status_code=422, detail=f"invalid item ids: {raw!r}")


def main():
    import uvicorn
    s = cfg.s
    uvicorn.run("cart_driven_recsys.api:app", host=s.api_host, port=s.api_port, workers=s.api_workers)


if __name__ == "__main__":
    main()