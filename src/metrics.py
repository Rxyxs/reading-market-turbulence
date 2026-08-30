"""RMSPE (Root Mean Squared Percentage Error) -- la metrica oficial de la
competencia Optiver, no MAE/RMSE plano: penaliza error relativo, apropiado
porque la volatilidad realizada varia en ordenes de magnitud entre
regimenes de mercado calmos y turbulentos."""

from __future__ import annotations

import numpy as np

EPS = 1e-6


def rmspe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.sqrt(np.mean(((y_true - y_pred) / (y_true + EPS)) ** 2)))
