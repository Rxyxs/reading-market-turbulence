"""Persistencia de features + metricas en DuckDB."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

DB_PATH = Path(__file__).resolve().parent.parent / "outputs" / "volatility.duckdb"


def export_results(features: pd.DataFrame, metrics: pd.DataFrame) -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(DB_PATH))
    con.execute("CREATE OR REPLACE TABLE features AS SELECT * FROM features")
    con.execute("CREATE OR REPLACE TABLE model_metrics AS SELECT * FROM metrics")
    con.close()
