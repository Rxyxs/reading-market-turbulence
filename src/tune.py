"""Busqueda de hiperparametros con Optuna para LightGBM, minimizando RMSPE
promedio en GroupKFold por dia -- el mismo protocolo del pipeline principal.
Nota honesta: el baseline de persistencia ya le gana a LightGBM sin afinar
(4.58 vs 5.56), asi que la pregunta real que responde este script es "el
tuning cierra la brecha con el baseline, o no alcanza" -- reportado tal
cual salga, no forzado.

    python -m src.tune
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import optuna
from lightgbm import LGBMRegressor
from sklearn.model_selection import GroupKFold

from src.ingest import load_all_trades, bucket_trades
from src.features import compute_bucket_features
from src.modeling import FEATURE_COLUMNS
from src.metrics import rmspe

ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = ROOT / "outputs" / "reports"
MODELS_DIR = ROOT / "outputs" / "models"

N_TRIALS = 30


def _objective(trial, X, y, groups):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600),
        "num_leaves": trial.suggest_int("num_leaves", 7, 63),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
    }
    gkf = GroupKFold(n_splits=5)
    scores = []
    for train_idx, test_idx in gkf.split(X, y, groups):
        model = LGBMRegressor(**params, random_state=42, verbose=-1)
        model.fit(X[train_idx], y[train_idx])
        pred = model.predict(X[test_idx])
        scores.append(rmspe(y[test_idx], pred))
    return float(np.mean(scores))


def main() -> None:
    print("[1/3] Cargando trades reales y features...")
    trades = load_all_trades()
    bucketed = bucket_trades(trades)
    table = compute_bucket_features(bucketed).to_pandas()

    X = table[FEATURE_COLUMNS].to_numpy()
    y = table["target_next_realized_volatility"].to_numpy()
    groups = table["day"].to_numpy()

    print(f"[2/3] Optuna: {N_TRIALS} trials, minimizando RMSPE promedio en GroupKFold x dia (5 folds)...")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(lambda t: _objective(t, X, y, groups), n_trials=N_TRIALS)

    print(f"\nMejor RMSPE (Optuna): {study.best_value:.4f}")
    print(f"Mejores parametros: {study.best_params}")

    print("[3/3] Reentrenando modelo final sobre todos los datos...")
    best_model = LGBMRegressor(**study.best_params, random_state=42, verbose=-1)
    best_model.fit(X, y)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODELS_DIR / "lightgbm_tuned.joblib")

    baseline_metrics = json.load(open(REPORTS_DIR / "model_metrics.json", encoding="utf-8"))
    baseline_lgbm_rmspe = next(m["rmspe_mean"] for m in baseline_metrics if m["model"] == "lightgbm")
    baseline_persistence_rmspe = next(m["rmspe_mean"] for m in baseline_metrics if m["model"] == "historical_baseline")

    result = {
        "baseline_lightgbm_rmspe": baseline_lgbm_rmspe,
        "tuned_lightgbm_rmspe": round(study.best_value, 4),
        "improvement": round(baseline_lgbm_rmspe - study.best_value, 4),
        "persistence_baseline_rmspe": baseline_persistence_rmspe,
        "tuned_beats_persistence_baseline": bool(study.best_value < baseline_persistence_rmspe),
        "n_trials": N_TRIALS,
        "best_params": study.best_params,
    }
    with open(REPORTS_DIR / "optuna_tuning_result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\n=== Resultado ===")
    print(f"LightGBM baseline (sin tuning): RMSPE={baseline_lgbm_rmspe}")
    print(f"LightGBM tuned (Optuna, {N_TRIALS} trials): RMSPE={study.best_value:.4f}")
    print(f"Baseline de persistencia: RMSPE={baseline_persistence_rmspe}")
    print(f"¿El LightGBM afinado supera al baseline de persistencia? {result['tuned_beats_persistence_baseline']}")
    print(f"\nGuardado en: {REPORTS_DIR / 'optuna_tuning_result.json'}")


if __name__ == "__main__":
    main()
