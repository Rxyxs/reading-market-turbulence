"""API FastAPI de scoring de volatilidad en tiempo real (modelo LightGBM).

    uvicorn src.api:app --reload

El modelo se carga de forma diferida y se re-carga solo si el archivo cambio en
disco, para que un re-entrenamiento quede servido sin reiniciar el proceso.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.modeling import FEATURE_COLUMNS

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "outputs" / "models"
MODEL_FILENAME = "lightgbm_model.joblib"

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Crypto Order-Flow Volatility Forecaster",
    version="1.1.0",
    description="Predice la volatilidad realizada del proximo bucket de 30s a partir del flujo de ordenes del actual.",
)

_model: Any | None = None
_model_key: tuple[str, float] | None = None


class BucketFeatures(BaseModel):
    """Features de un bucket de 30s. Los rangos son los que la propia construccion
    de features garantiza (ver src/features.py): un valor fuera de rango indica un
    error del cliente, no un caso extremo de mercado."""

    vwap: float = Field(gt=0, description="Precio promedio ponderado por volumen del bucket")
    order_flow_imbalance: float = Field(ge=-1, le=1, description="(compra - venta) / volumen total")
    price_range_pct: float = Field(ge=0, description="(maximo - minimo) / precio de apertura")
    bucket_return: float = Field(description="cierre / apertura - 1")
    realized_volatility: float = Field(ge=0, description="Raiz de la suma de retornos log al cuadrado")
    total_volume: float = Field(ge=0)
    dollar_volume: float = Field(ge=0)
    n_trades: int = Field(ge=1)
    taker_buy_volume: float = Field(ge=0)
    taker_sell_volume: float = Field(ge=0)


# El orden de las columnas es el contrato entre el modelo y la API: si alguien
# agrega una feature en src/modeling.py y no la agrega aca, el vector de entrada
# queda desalineado y el modelo predice sobre columnas corridas, en silencio.
_faltantes = set(FEATURE_COLUMNS) - set(BucketFeatures.model_fields)
_sobrantes = set(BucketFeatures.model_fields) - set(FEATURE_COLUMNS)
if _faltantes or _sobrantes:
    raise RuntimeError(
        "BucketFeatures no coincide con FEATURE_COLUMNS de src/modeling.py: "
        f"faltan {sorted(_faltantes)}, sobran {sorted(_sobrantes)}"
    )


class ScoreResponse(BaseModel):
    predicted_next_bucket_realized_volatility: float


class HealthResponse(BaseModel):
    status: str
    model_available: bool
    model_path: str
    n_features: int


def model_path() -> Path:
    """Se resuelve en cada llamada para que apuntar MODELS_DIR a otro directorio
    (tests, o un volumen montado) tenga efecto sin reimportar el modulo."""
    return MODELS_DIR / MODEL_FILENAME


def load_model() -> Any:
    """Devuelve el modelo, cargandolo si hace falta. Lanza HTTPException con un
    cuerpo JSON explicito si no existe (404) o si no se puede leer (500)."""
    global _model, _model_key

    ruta = model_path()
    if not ruta.is_file():
        _model, _model_key = None, None
        raise HTTPException(
            status_code=404,
            detail={
                "error": "modelo_no_encontrado",
                "mensaje": f"No existe el archivo del modelo en {ruta}.",
                "como_resolver": "Genera el modelo con 'python -m src.pipeline' o monta el volumen que lo contiene.",
            },
        )

    clave = (str(ruta), ruta.stat().st_mtime)
    if _model is None or _model_key != clave:
        try:
            _model = joblib.load(ruta)
        except Exception as error:  # archivo corrupto, truncado o de otra version
            _model, _model_key = None, None
            logger.exception("No se pudo cargar el modelo desde %s", ruta)
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "modelo_ilegible",
                    "mensaje": f"El archivo del modelo existe pero no se pudo cargar: {error}",
                    "como_resolver": "Regenera el modelo con 'python -m src.pipeline'.",
                },
            ) from error
        _model_key = clave
    return _model


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Nunca falla: reporta si el modelo esta disponible para que un orquestador
    distinga 'el proceso vive' de 'el proceso puede predecir'."""
    return HealthResponse(
        status="ok",
        model_available=model_path().is_file(),
        model_path=str(model_path()),
        n_features=len(FEATURE_COLUMNS),
    )


@app.post("/score", response_model=ScoreResponse)
def score(features: BucketFeatures) -> ScoreResponse:
    model = load_model()
    row = [[getattr(features, c) for c in FEATURE_COLUMNS]]
    try:
        pred = float(model.predict(row)[0])
    except Exception as error:
        logger.exception("Fallo la inferencia con el vector %s", row)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "fallo_de_inferencia",
                "mensaje": f"El modelo no pudo predecir sobre el vector recibido: {error}",
                "n_features_esperadas": len(FEATURE_COLUMNS),
            },
        ) from error

    return ScoreResponse(predicted_next_bucket_realized_volatility=round(pred, 8))
