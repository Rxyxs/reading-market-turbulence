"""Features de microestructura por bucket de 30s: VWAP, desbalance de flujo
de ordenes (OFI, via lado agresor real is_buyer_maker), volatilidad
realizada intra-bucket (misma definicion que Optiver: raiz de la suma de
retornos log al cuadrado, aplicada aqui sobre el precio de cada trade en
vez de un mid-price de libro), y el TARGET = volatilidad realizada del
bucket SIGUIENTE (forecasting real, sin fuga: features de t, target de
t+1)."""

from __future__ import annotations

import numpy as np
import polars as pl


def compute_bucket_features(bucketed: pl.DataFrame) -> pl.DataFrame:
    per_bucket = bucketed.sort("timestamp_ms").group_by(["symbol", "day", "bucket_start_ms"], maintain_order=True).agg(
        [
            pl.col("price").first().alias("open_price"),
            pl.col("price").last().alias("close_price"),
            pl.col("price").max().alias("high_price"),
            pl.col("price").min().alias("low_price"),
            (pl.col("price") * pl.col("quantity")).sum().alias("dollar_volume"),
            pl.col("quantity").sum().alias("total_volume"),
            pl.len().alias("n_trades"),
            pl.col("quantity").filter(pl.col("is_buyer_maker") == False).sum().alias("taker_buy_volume"),
            pl.col("quantity").filter(pl.col("is_buyer_maker") == True).sum().alias("taker_sell_volume"),
            pl.col("price").alias("price_path"),
        ]
    )

    per_bucket = per_bucket.with_columns(
        [
            (pl.col("dollar_volume") / pl.col("total_volume")).alias("vwap"),
            ((pl.col("taker_buy_volume") - pl.col("taker_sell_volume")) / pl.col("total_volume")).alias("order_flow_imbalance"),
            ((pl.col("high_price") - pl.col("low_price")) / pl.col("open_price")).alias("price_range_pct"),
            (pl.col("close_price") / pl.col("open_price") - 1.0).alias("bucket_return"),
        ]
    )

    def _realized_vol(prices: list[float]) -> float:
        arr = np.asarray(prices, dtype=np.float64)
        if len(arr) < 2:
            return 0.0
        log_returns = np.diff(np.log(arr))
        return float(np.sqrt(np.sum(log_returns**2)))

    realized_vol = per_bucket["price_path"].map_elements(_realized_vol, return_dtype=pl.Float64)
    per_bucket = per_bucket.with_columns(realized_vol.alias("realized_volatility")).drop("price_path")

    # target: volatilidad realizada del bucket SIGUIENTE, dentro del mismo dia
    # (nunca cruza el limite de dia, evita mezclar sesiones de mercado distintas)
    per_bucket = per_bucket.sort(["symbol", "day", "bucket_start_ms"])
    per_bucket = per_bucket.with_columns(
        pl.col("realized_volatility").shift(-1).over(["symbol", "day"]).alias("target_next_realized_volatility")
    )

    return per_bucket.drop_nulls(subset=["target_next_realized_volatility"])
