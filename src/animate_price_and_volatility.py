"""Genera la version animada (GIF) de outputs/reports/price_and_volatility.png.

Reutiliza exactamente los mismos datos reales (VWAP y volatilidad realizada por
bucket de 30s, BTCUSDT 2026-08-25) que produce el notebook
01_OrderFlow_EDA_and_Volatility.ipynb en la celda que genera price_and_volatility.png.
No se inventan valores: se sub-muestrea la serie ya calculada a ~45 frames para
mantener el GIF liviano.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import polars as pl

from src.ingest import COLUMNS, bucket_trades
from src.features import compute_bucket_features

REPORTS_DIR = Path("outputs/reports")
N_FRAMES = 45


def load_real_features() -> tuple:
    one_day = pl.read_csv(
        "data/raw_btc/BTCUSDT-aggTrades-2026-08-25.csv",
        has_header=False,
        new_columns=COLUMNS,
    )
    one_day = one_day.with_columns(
        (pl.col("timestamp_us") // 1_000).alias("timestamp_ms"),
        pl.lit("2026-08-25").alias("day"),
        pl.lit("BTCUSDT").alias("symbol"),
    )
    bucketed = bucket_trades(one_day)
    features = compute_bucket_features(bucketed).to_pandas()
    vwap = features["vwap"].to_numpy()
    vol = features["realized_volatility"].to_numpy()
    return vwap, vol


def subsample(arr: np.ndarray, n_frames: int) -> np.ndarray:
    """Elige n_frames indices (crecientes, terminando en el ultimo punto real)."""
    n = len(arr)
    if n <= n_frames:
        return np.arange(1, n + 1)
    idx = np.linspace(1, n, n_frames, dtype=int)
    idx = np.unique(idx)
    if idx[-1] != n:
        idx = np.append(idx, n)
    return idx


def make_animation(out_path: Path) -> None:
    vwap, vol = load_real_features()
    x = np.arange(len(vwap))
    frame_ends = subsample(vwap, N_FRAMES)

    with plt.style.context("dark_background"):
        fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        fig.patch.set_facecolor("black")

        line_price, = axes[0].plot([], [], color="#8AB4F8", linewidth=1.0)
        axes[0].set_xlim(0, len(vwap))
        axes[0].set_ylim(vwap.min() * 0.999, vwap.max() * 1.001)
        axes[0].set_title("VWAP por bucket de 30s — BTCUSDT, 2026-08-25")
        axes[0].set_ylabel("Precio (USDT)")

        line_vol, = axes[1].plot([], [], color="#F28B82", linewidth=1.0)
        axes[1].set_xlim(0, len(vol))
        axes[1].set_ylim(vol.min(), vol.max() * 1.05)
        axes[1].set_title("Volatilidad realizada por bucket de 30s")
        axes[1].set_ylabel("Vol. realizada")
        axes[1].set_xlabel("Bucket (orden temporal)")

        label_price = axes[0].annotate(
            "", xy=(0, vwap[0]), xytext=(15, 15), textcoords="offset points",
            color="white", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="#8AB4F8", ec="none", alpha=0.85),
        )
        label_vol = axes[1].annotate(
            "", xy=(0, vol[0]), xytext=(15, 15), textcoords="offset points",
            color="white", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="#F28B82", ec="none", alpha=0.85),
        )

        plt.tight_layout()

        def update(frame_idx):
            end = frame_ends[frame_idx]
            xs = x[:end]
            line_price.set_data(xs, vwap[:end])
            line_vol.set_data(xs, vol[:end])

            tip_x = xs[-1]
            price_val = vwap[end - 1]
            vol_val = vol[end - 1]

            label_price.xy = (tip_x, price_val)
            label_price.set_text(f"VWAP: {price_val:,.2f}")
            label_vol.xy = (tip_x, vol_val)
            label_vol.set_text(f"Vol. realizada: {vol_val:.2e}")

            return line_price, line_vol, label_price, label_vol

        ani = FuncAnimation(fig, update, frames=len(frame_ends), interval=120, blit=False)
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        ani.save(out_path, writer="pillow")
        plt.close(fig)


if __name__ == "__main__":
    make_animation(REPORTS_DIR / "price_and_volatility_animated.gif")
    print("Guardado:", REPORTS_DIR / "price_and_volatility_animated.gif")
