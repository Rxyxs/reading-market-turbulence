import duckdb
import pandas as pd
import pytest

from src.database import FEATURES_TABLE, METRICS_TABLE, export_results, read_table


def _features(n: int, vwap: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": ["BTCUSDT"] * n,
            "day": ["2026-08-25"] * n,
            "bucket_start_ms": [i * 30_000 for i in range(n)],
            "vwap": [vwap] * n,
            "target_next_realized_volatility": [0.001] * n,
        }
    )


def _metrics(rmspe: float) -> pd.DataFrame:
    return pd.DataFrame([{"model": "historical_baseline", "rmspe_mean": rmspe, "rmspe_std": 0.1}])


def test_export_crea_las_dos_tablas(tmp_path):
    db = tmp_path / "volatility.duckdb"
    export_results(_features(3, 100.0), _metrics(4.5), db_path=db)

    assert len(read_table(FEATURES_TABLE, db_path=db)) == 3
    assert read_table(METRICS_TABLE, db_path=db)["rmspe_mean"].tolist() == [4.5]


def test_corridas_consecutivas_actualizan_las_filas(tmp_path):
    """Regresion: la version anterior hacia CREATE OR REPLACE TABLE features AS
    SELECT * FROM features, que desde la segunda corrida recreaba la tabla a
    partir de si misma y dejaba los datos de la primera."""
    db = tmp_path / "volatility.duckdb"

    export_results(_features(3, 100.0), _metrics(4.5), db_path=db)
    export_results(_features(5, 200.0), _metrics(9.9), db_path=db)

    features = read_table(FEATURES_TABLE, db_path=db)
    assert len(features) == 5, "la segunda corrida no reemplazo las filas"
    assert features["vwap"].unique().tolist() == [200.0], "quedaron valores de la corrida anterior"
    assert read_table(METRICS_TABLE, db_path=db)["rmspe_mean"].tolist() == [9.9]


def test_la_tabla_sigue_al_esquema_del_dataframe(tmp_path):
    """Si el pipeline agrega o quita una feature, la tabla persistida la refleja."""
    db = tmp_path / "volatility.duckdb"
    export_results(_features(2, 100.0), _metrics(4.5), db_path=db)

    con_columna_nueva = _features(2, 100.0).assign(order_flow_imbalance=[0.1, -0.2])
    export_results(con_columna_nueva, _metrics(4.5), db_path=db)

    assert "order_flow_imbalance" in read_table(FEATURES_TABLE, db_path=db).columns


def test_no_quedan_vistas_temporales_registradas(tmp_path):
    """El DataFrame se registra solo para escribir: la base no debe quedar con vistas sueltas."""
    db = tmp_path / "volatility.duckdb"
    export_results(_features(2, 100.0), _metrics(4.5), db_path=db)

    con = duckdb.connect(str(db), read_only=True)
    try:
        nombres = {fila[0] for fila in con.execute("SELECT table_name FROM duckdb_tables()").fetchall()}
        vistas = {fila[0] for fila in con.execute("SELECT view_name FROM duckdb_views()").fetchall()}
    finally:
        con.close()

    assert nombres == {FEATURES_TABLE, METRICS_TABLE}
    assert not {v for v in vistas if v.startswith("_entrada_")}


def test_crea_el_directorio_de_la_base(tmp_path):
    db = tmp_path / "outputs" / "volatility.duckdb"
    export_results(_features(1, 100.0), _metrics(1.0), db_path=db)
    assert db.exists()


def test_leer_una_tabla_inexistente_falla_explicito(tmp_path):
    db = tmp_path / "volatility.duckdb"
    export_results(_features(1, 100.0), _metrics(1.0), db_path=db)
    with pytest.raises(duckdb.CatalogException):
        read_table("no_existe", db_path=db)
