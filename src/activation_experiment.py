"""Experimento adicional: compara la MLP de volatilidad (src/modeling.py)
entrenada con la loss custom RMSPE (rmspe_loss, diferenciable, optimiza
directamente la metrica de evaluacion en vez de MSE) bajo tres funciones de
activacion -- ReLU, GELU, Swish (SiLU) -- sobre el mismo esquema de
validacion GroupKFold por dia que el resto del proyecto.

No reemplaza el pipeline principal (src/pipeline.py): es un modulo aditivo
que reutiliza la tabla de features ya persistida en DuckDB (outputs/volatility.duckdb,
tabla `features`, generada por `python -m src.pipeline`) para no tener que
re-ingerir 24.8M trades crudos.

    python -m src.activation_experiment
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from src.metrics import rmspe
from src.modeling import FEATURE_COLUMNS, mlp_predict, rmspe_loss, train_mlp

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "outputs" / "volatility.duckdb"
REPORTS_DIR = ROOT / "outputs" / "reports"

ACTIVATIONS = ["relu", "gelu", "swish"]


def load_features_table() -> pd.DataFrame:
    con = duckdb.connect(str(DB_PATH), read_only=True)
    table = con.execute("SELECT * FROM features").df()
    con.close()
    return table


def run_activation_comparison(df: pd.DataFrame, n_splits: int = 5) -> pd.DataFrame:
    """Entrena la MLP con loss custom RMSPE para cada activacion, evaluada
    con GroupKFold por dia (mismo protocolo que evaluate_group_kfold)."""
    symbol_to_idx = {s: i for i, s in enumerate(sorted(df["symbol"].unique()))}
    gkf = GroupKFold(n_splits=n_splits)
    X_all = df[FEATURE_COLUMNS].to_numpy()
    y_all = df["target_next_realized_volatility"].to_numpy()
    groups = df["day"].to_numpy()
    symbols_idx = df["symbol"].map(symbol_to_idx).to_numpy()

    fold_scores = {act: [] for act in ACTIVATIONS}

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X_all, y_all, groups), start=1):
        for activation in ACTIVATIONS:
            model, scaler = train_mlp(
                X_all[train_idx], symbols_idx[train_idx], y_all[train_idx],
                X_all[test_idx], symbols_idx[test_idx], y_all[test_idx],
                n_symbols=len(symbol_to_idx),
                activation=activation, loss_fn=rmspe_loss,
            )
            pred = mlp_predict(model, scaler, X_all[test_idx], symbols_idx[test_idx])
            score = rmspe(y_all[test_idx], pred)
            fold_scores[activation].append(score)
        print(f"  fold {fold}/{n_splits}: " + " ".join(f"{a}={fold_scores[a][-1]:.4f}" for a in ACTIVATIONS))

    return pd.DataFrame(
        [{"activation": a, "loss": "rmspe_custom", "rmspe_mean": float(np.mean(v)), "rmspe_std": float(np.std(v))}
         for a, v in fold_scores.items()]
    ).sort_values("rmspe_mean")


def plot_comparison(results: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["#3b82f6", "#f59e0b", "#10b981"]
    ax.bar(results["activation"], results["rmspe_mean"], yerr=results["rmspe_std"],
           color=colors[: len(results)], capsize=4)
    ax.set_ylabel("RMSPE (media +/- std, GroupKFold x dia)")
    ax.set_title("MLP de volatilidad -- loss custom RMSPE por activacion")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    print("[1/3] Cargando tabla de features desde DuckDB (outputs/volatility.duckdb)...")
    df = load_features_table()
    print(f"  {len(df):,} buckets cargados")

    print("[2/3] Comparando activaciones (ReLU / GELU / Swish) con loss custom RMSPE, GroupKFold x dia...")
    results = run_activation_comparison(df)
    print("\n=== RMSPE por activacion (loss custom RMSPE) ===")
    print(results.to_string(index=False))

    print("\n[3/3] Guardando resultados...")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(REPORTS_DIR / "activation_comparison.csv", index=False)
    with open(REPORTS_DIR / "activation_comparison.json", "w", encoding="utf-8") as f:
        json.dump(results.to_dict(orient="records"), f, indent=2, ensure_ascii=False)
    plot_comparison(results, REPORTS_DIR / "activation_comparison.png")

    con = duckdb.connect(str(DB_PATH))
    con.execute("CREATE OR REPLACE TABLE activation_comparison AS SELECT * FROM results")
    con.close()

    print(f"Guardado en: {REPORTS_DIR}/activation_comparison.{{csv,json,png}}, tabla DuckDB activation_comparison")


if __name__ == "__main__":
    main()
