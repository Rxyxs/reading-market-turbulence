"""Exporta las features de un dia real (BTCUSDT, 2026-08-25) calculadas por
Python/Polars, como referencia para verificar go/streamer contra la misma
logica de bucketizacion de 30s.

    python -m src.export_for_go
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from src.ingest import COLUMNS, bucket_trades
from src.features import compute_bucket_features

ROOT = Path(__file__).resolve().parent.parent
GO_DIR = ROOT / "go"
REFERENCE_DAY = "2026-08-25"


def main() -> None:
    print(f"[1/2] Cargando trades reales BTCUSDT del {REFERENCE_DAY}...")
    trades = pl.read_csv(
        ROOT / "data" / "raw_btc" / f"BTCUSDT-aggTrades-{REFERENCE_DAY}.csv",
        has_header=False,
        new_columns=COLUMNS,
    )
    trades = trades.with_columns(
        (pl.col("timestamp_us") // 1_000).alias("timestamp_ms"),
        pl.lit(REFERENCE_DAY).alias("day"),
        pl.lit("BTCUSDT").alias("symbol"),
    )
    print(f"  {trades.height:,} trades reales cargados")

    print("[2/2] Calculando features de referencia (buckets de 30s)...")
    bucketed = bucket_trades(trades)
    features = compute_bucket_features(bucketed)

    GO_DIR.mkdir(parents=True, exist_ok=True)
    features.select(
        ["bucket_start_ms", "vwap", "order_flow_imbalance", "realized_volatility", "n_trades", "total_volume"]
    ).write_csv(GO_DIR / "python_reference.csv")

    print(f"  {features.height} buckets calculados")
    print(f"\nGuardado en: {GO_DIR}/python_reference.csv")


if __name__ == "__main__":
    main()
