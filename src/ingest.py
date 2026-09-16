"""Ingesta de trades reales (Binance data.vision, no requiere API key): dumps
diarios de aggTrades tick-by-tick.

No es el libro de ordenes L2 completo de Optiver (bid/ask por nivel) --
Binance no publica dumps historicos de profundidad L2 gratis. Se usa el
flujo de trades real (agresor comprador/vendedor, precio, volumen, timestamp
al microsegundo) para construir features de microestructura genuinas
(VWAP, desbalance de flujo de ordenes via lado del agresor, volatilidad
realizada intra-dia) -- real, pero de una granularidad de dato distinta a
la del dataset original de la competencia. Disclosure honesto, no oculto.

Los archivos se bajan con `python scripts/download.py` y quedan en
data/raw/<SIMBOLO>/. Tambien se acepta el layout viejo (data/raw_btc,
data/raw_eth) para no invalidar descargas previas.
"""

from __future__ import annotations

import re
from pathlib import Path

import polars as pl

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"
RAW_ROOT = DATA_ROOT / "raw"

SYMBOLS = ("BTCUSDT", "ETHUSDT")

# Layout anterior, previo a scripts/download.py.
LEGACY_DIRS = {"BTCUSDT": DATA_ROOT / "raw_btc", "ETHUSDT": DATA_ROOT / "raw_eth"}
SYMBOL_DIRS = LEGACY_DIRS  # compatibilidad con codigo que lo importaba

COLUMNS = ["agg_trade_id", "price", "quantity", "first_trade_id", "last_trade_id",
           "timestamp_us", "is_buyer_maker", "is_best_match"]

BUCKET_SECONDS = 30

PATRON_DIA = re.compile(r"(\d{4}-\d{2}-\d{2})")


def day_from_path(path: Path) -> str:
    """Extrae el dia del nombre del archivo (BTCUSDT-aggTrades-2026-08-25.csv)."""
    encontrado = PATRON_DIA.search(path.stem)
    if not encontrado:
        raise ValueError(f"no se pudo deducir el dia del nombre del archivo: {path.name}")
    return encontrado.group(1)


def symbol_files(symbol: str) -> list[Path]:
    """Archivos diarios de un simbolo, en cualquiera de los dos layouts, sin duplicar."""
    encontrados: dict[str, Path] = {}
    for directorio in (RAW_ROOT / symbol, LEGACY_DIRS.get(symbol)):
        if directorio is None or not directorio.is_dir():
            continue
        for archivo in sorted(directorio.glob("*.csv")):
            encontrados.setdefault(archivo.name, archivo)
    return [encontrados[nombre] for nombre in sorted(encontrados)]


def day_file(symbol: str, day: str) -> Path:
    """Ruta del archivo de un dia concreto. Lanza FileNotFoundError si no esta descargado."""
    for archivo in symbol_files(symbol):
        if day_from_path(archivo) == day:
            return archivo
    raise FileNotFoundError(
        f"no hay datos de {symbol} para {day}. Descargalos con: "
        f"python scripts/download.py --symbols {symbol} --start {day} --end {day}"
    )


def load_all_trades(symbols: tuple[str, ...] = SYMBOLS) -> pl.DataFrame:
    """Carga todos los dias descargados de cada simbolo."""
    frames = []
    for symbol in symbols:
        for archivo in symbol_files(symbol):
            df = pl.read_csv(archivo, has_header=False, new_columns=COLUMNS)
            df = df.with_columns(
                pl.lit(day_from_path(archivo)).alias("day"),
                pl.lit(symbol).alias("symbol"),
            )
            frames.append(df)

    if not frames:
        raise FileNotFoundError(
            f"no se encontraron archivos de trades en {RAW_ROOT} para {', '.join(symbols)}. "
            "Descargalos con: python scripts/download.py --symbols "
            f"{' '.join(symbols)} --start YYYY-MM-DD --end YYYY-MM-DD"
        )

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
