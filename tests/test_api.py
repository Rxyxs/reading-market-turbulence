import joblib
import numpy as np
import pytest
from fastapi.testclient import TestClient
from lightgbm import LGBMRegressor

import src.api as api
from src.modeling import FEATURE_COLUMNS

VECTOR_VALIDO = {
    "vwap": 111_000.0,
    "order_flow_imbalance": -0.12,
    "price_range_pct": 0.0004,
    "bucket_return": -0.0001,
    "realized_volatility": 0.00031,
    "total_volume": 12.5,
    "dollar_volume": 1_387_500.0,
    "n_trades": 842,
    "taker_buy_volume": 5.5,
    "taker_sell_volume": 7.0,
}


@pytest.fixture
def modelos(tmp_path, monkeypatch):
    """Apunta la API a un directorio de modelos vacio y limpia el cache del proceso."""
    monkeypatch.setattr(api, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(api, "_model", None)
    monkeypatch.setattr(api, "_model_key", None)
    return tmp_path


@pytest.fixture
def cliente(modelos):
    return TestClient(api.app)


def entrenar_modelo(destino):
    """Modelo real minimo (mismas 10 features que el pipeline), suficiente para servir."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, len(FEATURE_COLUMNS)))
    y = np.abs(rng.normal(size=80)) + 0.1
    modelo = LGBMRegressor(n_estimators=5, num_leaves=3, min_child_samples=5, verbose=-1).fit(X, y)
    joblib.dump(modelo, destino)
    return modelo


def test_health_responde_sin_modelo(cliente):
    respuesta = cliente.get("/health")
    assert respuesta.status_code == 200
    cuerpo = respuesta.json()
    assert cuerpo["status"] == "ok"
    assert cuerpo["model_available"] is False
    assert cuerpo["n_features"] == len(FEATURE_COLUMNS)


def test_health_reporta_el_modelo_cuando_existe(cliente, modelos):
    entrenar_modelo(modelos / api.MODEL_FILENAME)
    assert cliente.get("/health").json()["model_available"] is True


def test_score_sin_modelo_devuelve_404_con_json_claro(cliente):
    respuesta = cliente.post("/score", json=VECTOR_VALIDO)
    assert respuesta.status_code == 404
    detalle = respuesta.json()["detail"]
    assert detalle["error"] == "modelo_no_encontrado"
    assert "src.pipeline" in detalle["como_resolver"]


def test_score_con_modelo_devuelve_una_prediccion_finita(cliente, modelos):
    entrenar_modelo(modelos / api.MODEL_FILENAME)
    respuesta = cliente.post("/score", json=VECTOR_VALIDO)
    assert respuesta.status_code == 200
    valor = respuesta.json()["predicted_next_bucket_realized_volatility"]
    assert isinstance(valor, float)
    assert np.isfinite(valor)


def test_modelo_corrupto_devuelve_500(cliente, modelos):
    (modelos / api.MODEL_FILENAME).write_bytes(b"esto no es un joblib")
    respuesta = cliente.post("/score", json=VECTOR_VALIDO)
    assert respuesta.status_code == 500
    assert respuesta.json()["detail"]["error"] == "modelo_ilegible"


def test_fallo_de_inferencia_devuelve_500(cliente, modelos, monkeypatch):
    entrenar_modelo(modelos / api.MODEL_FILENAME)
    assert cliente.post("/score", json=VECTOR_VALIDO).status_code == 200

    class ModeloRoto:
        def predict(self, X):
            raise ValueError("numero de features inesperado")

    monkeypatch.setattr(api, "_model", ModeloRoto())
    respuesta = cliente.post("/score", json=VECTOR_VALIDO)
    assert respuesta.status_code == 500
    assert respuesta.json()["detail"]["error"] == "fallo_de_inferencia"


def test_el_modelo_se_recarga_si_cambia_en_disco(cliente, modelos):
    """Un re-entrenamiento debe quedar servido sin reiniciar el proceso."""
    ruta = modelos / api.MODEL_FILENAME
    entrenar_modelo(ruta)
    primera = cliente.post("/score", json=VECTOR_VALIDO).json()

    rng = np.random.default_rng(7)
    X = rng.normal(size=(80, len(FEATURE_COLUMNS)))
    otro = LGBMRegressor(n_estimators=5, num_leaves=3, min_child_samples=5, verbose=-1)
    otro.fit(X, np.abs(rng.normal(size=80)) * 100 + 50)
    joblib.dump(otro, ruta)
    import os

    os.utime(ruta, (ruta.stat().st_atime, ruta.stat().st_mtime + 10))

    segunda = cliente.post("/score", json=VECTOR_VALIDO).json()
    assert segunda != primera


@pytest.mark.parametrize(
    "campo, valor",
    [
        ("n_trades", 0),
        ("order_flow_imbalance", 1.5),
        ("vwap", 0.0),
        ("realized_volatility", -0.1),
        ("total_volume", -1.0),
    ],
)
def test_valores_fuera_de_rango_devuelven_422(cliente, campo, valor):
    respuesta = cliente.post("/score", json={**VECTOR_VALIDO, campo: valor})
    assert respuesta.status_code == 422
    assert respuesta.json()["detail"][0]["loc"][-1] == campo


def test_campo_faltante_devuelve_422(cliente):
    payload = {k: v for k, v in VECTOR_VALIDO.items() if k != "vwap"}
    respuesta = cliente.post("/score", json=payload)
    assert respuesta.status_code == 422
    assert respuesta.json()["detail"][0]["loc"][-1] == "vwap"


def test_el_contrato_de_features_coincide_con_el_modelo():
    assert set(api.BucketFeatures.model_fields) == set(FEATURE_COLUMNS)
