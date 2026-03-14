from __future__ import annotations

from functools import lru_cache

import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field

from cart_driven_recsys.config import cfg
from cart_driven_recsys.recommenders import recommend_with_hybrid


app = FastAPI(title="cart-driven-recsys")


class CartRecommendationRequest(BaseModel):
    item_ids: list[int] = Field(default_factory=list)
    k: int = Field(default=10, ge=1, le=100)


class CartRecommendationResponse(BaseModel):
    item_ids: list[int]
    model_type: str


@lru_cache(maxsize=1)
def load_artifact() -> dict:
    artifact = joblib.load(cfg.model_artifact)
    artifact["item_ids"] = np.asarray(artifact["item_ids"])
    artifact["item_id_to_index"] = {
        int(item_id): item_index
        for item_index, item_id in enumerate(artifact["item_ids"])
    }
    return artifact


@app.get("/health")
def health() -> dict[str, str]:
    artifact = load_artifact()
    return {"status": "ok", "model_type": str(artifact["model_type"])}


@app.post("/recommendations/cart", response_model=CartRecommendationResponse)
def recommend_cart(request: CartRecommendationRequest) -> CartRecommendationResponse:
    artifact = load_artifact()

    recommended_item_ids = recommend_with_hybrid(
        model=artifact["als_model"],
        item_ids=artifact["item_ids"],
        item_id_to_index=artifact["item_id_to_index"],
        covisit_index=artifact["covisit_index"],
        popular_items=artifact["popular_items"],
        cart_item_ids=request.item_ids,
        k=request.k,
        als_weight=float(artifact["hybrid_params"]["als_weight"]),
        covisit_weight=float(artifact["hybrid_params"]["covisit_weight"]),
        rrf_constant=int(artifact["hybrid_params"]["rrf_constant"]),
    )

    return CartRecommendationResponse(
        item_ids=recommended_item_ids,
        model_type=str(artifact["model_type"]),
    )