"""Persistencia de features + metricas en DuckDB."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

DB_PATH = Path(__file__).resolve().parent.parent / "outputs" / "volatility.duckdb"

FEATURES_TABLE = "features"
METRICS_TABLE = "model_metrics"


def write_table(con: duckdb.DuckDBPyConnection, table: str, df: pd.DataFrame) -> None:
    """Reemplaza `table` con el contenido de `df`.

    El DataFrame se registra bajo un nombre propio antes de la consulta. Sin eso,
    `CREATE OR REPLACE TABLE features AS SELECT * FROM features` resuelve `features`
    contra la tabla ya existente en vez del DataFrame: la primera corrida escribe
    bien y todas las siguientes recrean la tabla a partir de si misma, conservando
    datos viejos sin emitir ningun error.
    """
    vista = f"_entrada_{table}"
    con.register(vista, df)
    try:
        con.execute(f'CREATE OR REPLACE TABLE "{table}" AS SELECT * FROM "{vista}"')
    finally:
        con.unregister(vista)


def export_results(
    features: pd.DataFrame,
    metrics: pd.DataFrame,
    db_path: Path | str = DB_PATH,
) -> Path:
    """Persiste la tabla de features y las metricas por modelo. Devuelve la ruta de la base."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    try:
        write_table(con, FEATURES_TABLE, features)
        write_table(con, METRICS_TABLE, metrics)
    finally:
        con.close()
    return db_path


def read_table(table: str, db_path: Path | str = DB_PATH) -> pd.DataFrame:
    """Lee una tabla persistida, sin re-correr el pipeline."""
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        return con.execute(f'SELECT * FROM "{table}"').df()
    finally:
        con.close()
