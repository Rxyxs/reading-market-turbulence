"""Ingesta de trades reales BTC/USDT (Binance data.vision, no requiere API
key): 10 dias completos de aggTrades tick-by-tick, ~1.1GB crudos.

No es el libro de ordenes L2 completo de Optiver (bid/ask por nivel) --
Binance no publica dumps historicos de profundidad L2 gratis. Se usa el
flujo de trades real (agresor comprador/vendedor, precio, volumen, timestamp
al microsegundo) para construir features de microestructura genuinas
(VWAP, desbalance de flujo de ordenes via lado del agresor, volatilidad
realizada intra-dia) -- real, pero de una granularidad de dato distinta a
la del dataset original de la competencia. Disclosure honesto, no oculto.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"

SYMBOL_DIRS = {"BTCUSDT": DATA_ROOT / "raw_btc", "ETHUSDT": DATA_ROOT / "raw_eth"}

COLUMNS = ["agg_trade_id", "price", "quantity", "first_trade_id", "last_trade_id",
           "timestamp_us", "is_buyer_maker", "is_best_match"]

BUCKET_SECONDS = 30


def load_all_trades() -> pl.DataFrame:
    """Carga los 2 activos (BTCUSDT, ETHUSDT), 10 dias reales cada uno."""
    frames = []
    for symbol, directory in SYMBOL_DIRS.items():
        for f in sorted(directory.glob("*.csv")):
            df = pl.read_csv(f, has_header=False, new_columns=COLUMNS)
            day = f.stem.split("-")[-3] + "-" + f.stem.split("-")[-2] + "-" + f.stem.split("-")[-1]
            df = df.with_columns(pl.lit(day).alias("day"), pl.lit(symbol).alias("symbol"))
            frames.append(df)
    trades = pl.concat(frames)
    trades = trades.with_columns((pl.col("timestamp_us") // 1_000).alias("timestamp_ms"))
    return trades.sort(["symbol", "timestamp_ms"])


def bucket_trades(trades: pl.DataFrame, bucket_seconds: int = BUCKET_SECONDS) -> pl.DataFrame:
    """Agrupa trades en buckets de tiempo fijos (30s por defecto) por dia --
    el equivalente de `time_id` de Optiver, pero derivado de tiempo real de
    reloj, no de un identificador anonimizado."""
    bucket_ms = bucket_seconds * 1_000
    return trades.with_columns(
        (pl.col("timestamp_ms") // bucket_ms * bucket_ms).alias("bucket_start_ms")
    )
