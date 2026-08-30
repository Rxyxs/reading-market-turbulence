"""Iteracion de modelos: baseline historico (persistencia) -> LightGBM ->
MLP tabular en PyTorch con embedding de activo (BTCUSDT/ETHUSDT), optimizado
para RMSPE. GroupKFold por dia (10 dias reales) evita fuga temporal."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from lightgbm import LGBMRegressor
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from src.metrics import rmspe

RANDOM_STATE = 42

FEATURE_COLUMNS = [
    "vwap", "order_flow_imbalance", "price_range_pct", "bucket_return",
    "realized_volatility", "total_volume", "dollar_volume", "n_trades",
    "taker_buy_volume", "taker_sell_volume",
]


def historical_baseline_predict(df: pd.DataFrame) -> np.ndarray:
    """Baseline 'historico': la volatilidad del proximo bucket = la del bucket actual
    (persistencia). Estandar en forecasting de volatilidad de muy corto plazo."""
    return df["realized_volatility"].to_numpy()


def train_lightgbm(X_train, y_train) -> LGBMRegressor:
    model = LGBMRegressor(
        n_estimators=400, num_leaves=31, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE, verbose=-1,
    )
    model.fit(X_train, y_train)
    return model


class VolatilityMLP(nn.Module):
    """MLP tabular con embedding de activo -- permite que BTCUSDT/ETHUSDT
    compartan la misma red pero aprendan un offset especifico de regimen de
    volatilidad por activo, en vez de una red por activo."""

    def __init__(self, n_numeric_features: int, n_symbols: int, embedding_dim: int = 4, hidden: tuple[int, ...] = (64, 32)):
        super().__init__()
        self.embedding = nn.Embedding(n_symbols, embedding_dim)
        prev = n_numeric_features + embedding_dim
        layers = []
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(0.2)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x_numeric, symbol_idx):
        emb = self.embedding(symbol_idx)
        x = torch.cat([x_numeric, emb], dim=1)
        return self.net(x).squeeze(-1)


def train_mlp(
    X_train: np.ndarray, symbols_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, symbols_val: np.ndarray, y_val: np.ndarray,
    n_symbols: int, epochs: int = 60, batch_size: int = 512, lr: float = 1e-3, patience: int = 8,
) -> tuple[VolatilityMLP, StandardScaler]:
    scaler = StandardScaler().fit(X_train)
    Xt = torch.tensor(scaler.transform(X_train), dtype=torch.float32)
    st = torch.tensor(symbols_train, dtype=torch.long)
    yt = torch.tensor(y_train, dtype=torch.float32)
    Xv = torch.tensor(scaler.transform(X_val), dtype=torch.float32)
    sv = torch.tensor(symbols_val, dtype=torch.long)
    yv = torch.tensor(y_val, dtype=torch.float32)

    model = VolatilityMLP(n_numeric_features=X_train.shape[1], n_symbols=n_symbols)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = nn.MSELoss()

    n = Xt.shape[0]
    best_val, best_state, bad = float("inf"), None, 0
    rng = np.random.default_rng(RANDOM_STATE)

    for epoch in range(epochs):
        model.train()
        perm = rng.permutation(n)
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            optimizer.zero_grad()
            loss = loss_fn(model(Xt[idx], st[idx]), yt[idx])
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(Xv, sv), yv).item()
        if val_loss < best_val:
            best_val, best_state, bad = val_loss, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break

    model.load_state_dict(best_state)
    return model, scaler


def mlp_predict(model: VolatilityMLP, scaler: StandardScaler, X: np.ndarray, symbols: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(scaler.transform(X), dtype=torch.float32), torch.tensor(symbols, dtype=torch.long))
        return pred.numpy()


def evaluate_group_kfold(df: pd.DataFrame, symbol_to_idx: dict, n_splits: int = 5) -> pd.DataFrame:
    gkf = GroupKFold(n_splits=n_splits)
    X_all = df[FEATURE_COLUMNS].to_numpy()
    y_all = df["target_next_realized_volatility"].to_numpy()
    groups = df["day"].to_numpy()
    symbols_idx = df["symbol"].map(symbol_to_idx).to_numpy()

    fold_scores = {"historical_baseline": [], "lightgbm": [], "pytorch_mlp_embeddings": []}

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X_all, y_all, groups), start=1):
        df_test = df.iloc[test_idx]
        baseline_pred = historical_baseline_predict(df_test)
        fold_scores["historical_baseline"].append(rmspe(y_all[test_idx], baseline_pred))

        lgbm = train_lightgbm(X_all[train_idx], y_all[train_idx])
        lgbm_pred = lgbm.predict(X_all[test_idx])
        fold_scores["lightgbm"].append(rmspe(y_all[test_idx], lgbm_pred))

        mlp, scaler = train_mlp(
            X_all[train_idx], symbols_idx[train_idx], y_all[train_idx],
            X_all[test_idx], symbols_idx[test_idx], y_all[test_idx],
            n_symbols=len(symbol_to_idx),
        )
        mlp_pred = mlp_predict(mlp, scaler, X_all[test_idx], symbols_idx[test_idx])
        fold_scores["pytorch_mlp_embeddings"].append(rmspe(y_all[test_idx], mlp_pred))

        print(f"  fold {fold}/{n_splits} (dia held-out): baseline={fold_scores['historical_baseline'][-1]:.4f} "
              f"lgbm={fold_scores['lightgbm'][-1]:.4f} mlp={fold_scores['pytorch_mlp_embeddings'][-1]:.4f}")

    return pd.DataFrame(
        [{"model": name, "rmspe_mean": float(np.mean(v)), "rmspe_std": float(np.std(v))} for name, v in fold_scores.items()]
    ).sort_values("rmspe_mean")
