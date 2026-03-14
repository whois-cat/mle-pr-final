from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from cart_driven_recsys.config import cfg
from cart_driven_recsys.recommenders import recommend_with_hybrid


logger = logging.getLogger(__name__)


app = FastAPI(
    title="cart-driven-recsys",
    version="1.0.0",
    summary="cart recommendation service",
    description="ALS + co-visitation + popularity fallback",
)


class HealthResponse(BaseModel):
    status: str


class MetadataResponse(BaseModel):
    model_type: str
    artifact_path: str
    n_items: int
    n_users: int
    cutoff_date: str
    als_factors: int
    covisit_top_neighbors: int
    als_weight: float
    covisit_weight: float
    rrf_constant: int


class CartRecommendationRequest(BaseModel):
    item_ids: list[int] = Field(default_factory=list, description="item ids currently in cart")
    k: int = Field(default=10, ge=1, le=100, description="number of recommendations to return")


class CartRecommendationResponse(BaseModel):
    requested_k: int
    cart_item_ids: list[int]
    unknown_item_ids: list[int]
    item_ids: list[int]


@lru_cache(maxsize=1)
def _load_artifact() -> dict[str, Any]:
    artifact = joblib.load(cfg.model_artifact)

    required_keys = {
        "model_type",
        "als_model",
        "item_ids",
        "popular_items",
        "covisit_index",
        "hybrid_params",
        "train_meta",
    }

    missing_keys = required_keys - set(artifact.keys())
    if missing_keys:
        raise RuntimeError(f"artifact is missing required keys: {sorted(missing_keys)}")

    artifact["item_ids"] = np.asarray(artifact["item_ids"], dtype=np.int64)
    artifact["item_id_to_index"] = {
        int(item_id): item_index
        for item_index, item_id in enumerate(artifact["item_ids"])
    }
    artifact["item_id_set"] = set(int(item_id) for item_id in artifact["item_ids"])

    return artifact


def get_artifact() -> dict[str, Any]:
    try:
        return _load_artifact()
    except FileNotFoundError as error:
        logger.exception("model artifact not found")
        raise HTTPException(status_code=500, detail="model artifact not found") from error
    except Exception as error:
        logger.exception("failed to load model artifact")
        raise HTTPException(status_code=500, detail="failed to load model artifact") from error


def build_metadata_response(artifact: dict[str, Any]) -> MetadataResponse:
    train_meta = artifact["train_meta"]
    hybrid_params = artifact["hybrid_params"]

    return MetadataResponse(
        model_type=str(artifact["model_type"]),
        artifact_path=str(cfg.model_artifact),
        n_items=int(train_meta["n_items"]),
        n_users=int(train_meta["n_users"]),
        cutoff_date=str(train_meta["cutoff_date"]),
        als_factors=int(train_meta["als_factors"]),
        covisit_top_neighbors=int(train_meta["covisit_top_neighbors"]),
        als_weight=float(hybrid_params["als_weight"]),
        covisit_weight=float(hybrid_params["covisit_weight"]),
        rrf_constant=int(hybrid_params["rrf_constant"]),
    )


def normalize_cart_item_ids(item_ids: list[int]) -> list[int]:
    return list(dict.fromkeys(int(item_id) for item_id in item_ids))


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    get_artifact()
    return HealthResponse(status="ok")


@app.get("/metadata", response_model=MetadataResponse)
def metadata() -> MetadataResponse:
    artifact = get_artifact()
    return build_metadata_response(artifact)


@app.post("/recommendations/cart", response_model=CartRecommendationResponse)
def recommend_cart(request: CartRecommendationRequest) -> CartRecommendationResponse:
    artifact = get_artifact()

    cart_item_ids = normalize_cart_item_ids(request.item_ids)
    unknown_item_ids = [
        item_id
        for item_id in cart_item_ids
        if item_id not in artifact["item_id_set"]
    ]

    recommended_item_ids = recommend_with_hybrid(
        model=artifact["als_model"],
        item_ids=artifact["item_ids"],
        item_id_to_index=artifact["item_id_to_index"],
        covisit_index=artifact["covisit_index"],
        popular_items=artifact["popular_items"],
        cart_item_ids=cart_item_ids,
        k=request.k,
        als_weight=float(artifact["hybrid_params"]["als_weight"]),
        covisit_weight=float(artifact["hybrid_params"]["covisit_weight"]),
        rrf_constant=int(artifact["hybrid_params"]["rrf_constant"]),
    )

    return CartRecommendationResponse(
        requested_k=request.k,
        cart_item_ids=cart_item_ids,
        unknown_item_ids=unknown_item_ids,
        item_ids=recommended_item_ids,
    )