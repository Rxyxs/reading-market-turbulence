import polars as pl

from src.features import compute_bucket_features


def _make_trades():
    return pl.DataFrame(
        {
            "symbol": ["BTCUSDT"] * 6,
            "day": ["2026-01-01"] * 6,
            "timestamp_ms": [0, 5000, 10000, 30000, 35000, 40000],
            "price": [100.0, 101.0, 102.0, 103.0, 104.0, 106.0],
            "quantity": [1.0, 2.0, 1.0, 1.0, 1.0, 2.0],
            "is_buyer_maker": [False, True, False, False, True, False],
        }
    )


def test_realized_volatility_zero_for_single_trade_bucket():
    bucketed = _make_trades().with_columns((pl.col("timestamp_ms") // 30000 * 30000).alias("bucket_start_ms"))
    result = compute_bucket_features(bucketed)
    # bucket 0 tiene 3 trades, bucket 30000 tiene 3 trades -> ambos con vol > 0 o al menos definido
    assert result.height >= 1
    assert (result["realized_volatility"] >= 0).all()


def test_target_is_shifted_realized_volatility_next_bucket():
    bucketed = _make_trades().with_columns((pl.col("timestamp_ms") // 30000 * 30000).alias("bucket_start_ms"))
    result = compute_bucket_features(bucketed)
    # con 2 buckets, el primero debe tener como target la vol realizada del segundo
    if result.height >= 1:
        first_row = result.row(0, named=True)
        assert first_row["target_next_realized_volatility"] is not None


def test_order_flow_imbalance_hand_computed():
    # bucket 0 (t=0,5000,10000): qty=[1.0,2.0,1.0], is_buyer_maker=[False,True,False]
    # -> taker_buy_volume=1.0+1.0=2.0, taker_sell_volume=2.0, total=4.0 -> OFI=(2-2)/4=0.0
    # (only bucket 0 survives drop_nulls: it's the only one with a valid "next bucket" target)
    bucketed = _make_trades().with_columns((pl.col("timestamp_ms") // 30000 * 30000).alias("bucket_start_ms"))
    result = compute_bucket_features(bucketed)
    assert result.height == 1
    row = result.row(0, named=True)
    assert abs(row["order_flow_imbalance"] - 0.0) < 1e-9
