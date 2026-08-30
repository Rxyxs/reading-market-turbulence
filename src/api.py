"""API FastAPI de scoring de volatilidad en tiempo real (modelo LightGBM).

    uvicorn src.api:app --reload
"""

from __future__ import annotations

from pathlib import Path

import joblib
from fastapi import FastAPI
from pydantic import BaseModel

from src.modeling import FEATURE_COLUMNS

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "outputs" / "models"

app = FastAPI(title="Crypto Order-Flow Volatility Forecaster", version="1.0.0")

_model = None


def _lazy_load():
    global _model
    if _model is None:
        _model = joblib.load(MODELS_DIR / "lightgbm_model.joblib")


class BucketFeatures(BaseModel):
    vwap: float
    order_flow_imbalance: float
    price_range_pct: float
    bucket_return: float
    realized_volatility: float
    total_volume: float
    dollar_volume: float
    n_trades: int
    taker_buy_volume: float
    taker_sell_volume: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/score")
def score(features: BucketFeatures):
    _lazy_load()
    row = [[getattr(features, c) for c in FEATURE_COLUMNS]]
    pred = float(_model.predict(row)[0])
    return {"predicted_next_bucket_realized_volatility": round(pred, 8)}
