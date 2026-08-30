"""Pipeline end-to-end: ingesta (trades reales Binance) -> buckets 30s ->
features de microestructura -> iteracion de modelos -> GroupKFold por dia
-> DuckDB.

    python -m src.pipeline
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from src.ingest import load_all_trades, bucket_trades
from src.features import compute_bucket_features
from src.modeling import FEATURE_COLUMNS, evaluate_group_kfold, train_lightgbm, train_mlp
from src.database import export_results

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "outputs" / "models"
REPORTS_DIR = ROOT / "outputs" / "reports"


def main() -> None:
    print("[1/5] Cargando trades reales BTCUSDT + ETHUSDT (Binance, 10 dias c/u)...")
    trades = load_all_trades()
    print(f"  {trades.height:,} trades reales cargados")

    print("[2/5] Bucketizando en ventanas de 30s...")
    bucketed = bucket_trades(trades)

    print("[3/5] Feature engineering (VWAP, order flow imbalance, volatilidad realizada)...")
    table = compute_bucket_features(bucketed).to_pandas()
    print(f"  {len(table):,} buckets con target valido")

    symbol_to_idx = {s: i for i, s in enumerate(sorted(table["symbol"].unique()))}

    print("[4/5] Iteracion: baseline historico -> LightGBM -> PyTorch MLP con embeddings de activo (GroupKFold x dia)...")
    results = evaluate_group_kfold(table, symbol_to_idx, n_splits=5)
    print("\n=== RMSPE por modelo (promedio 5 folds, GroupKFold por dia) ===")
    print(results.to_string(index=False))

    print("\n[5/5] Ajustando LightGBM final (mejor RMSPE) sobre todos los datos...")
    X_all = table[FEATURE_COLUMNS].to_numpy()
    y_all = table["target_next_realized_volatility"].to_numpy()
    final_model = train_lightgbm(X_all, y_all)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, MODELS_DIR / "lightgbm_model.joblib")
    joblib.dump(symbol_to_idx, MODELS_DIR / "symbol_to_idx.joblib")

    results.to_csv(REPORTS_DIR / "model_metrics.csv", index=False)
    with open(REPORTS_DIR / "model_metrics.json", "w", encoding="utf-8") as f:
        json.dump(results.to_dict(orient="records"), f, indent=2, ensure_ascii=False)

    export_results(table, results)
    print(f"\nGuardado en: {MODELS_DIR}, {REPORTS_DIR}, outputs/volatility.duckdb")


if __name__ == "__main__":
    main()
