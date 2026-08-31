import numpy as np
import torch

from src.modeling import VolatilityMLP, rmspe_loss, train_mlp, mlp_predict


def test_rmspe_loss_zero_for_perfect_prediction():
    y = torch.tensor([0.001, 0.002, 0.0005])
    assert rmspe_loss(y, y).item() < 1e-6


def test_rmspe_loss_matches_numpy_rmspe():
    from src.metrics import rmspe

    y_true = np.array([1.0, 2.0, 0.5])
    y_pred = np.array([1.1, 1.8, 0.6])
    expected = rmspe(y_true, y_pred)
    actual = rmspe_loss(torch.tensor(y_pred), torch.tensor(y_true)).item()
    assert abs(expected - actual) < 1e-6


def test_volatility_mlp_supports_relu_gelu_swish():
    for activation in ("relu", "gelu", "swish"):
        model = VolatilityMLP(n_numeric_features=5, n_symbols=2, activation=activation)
        x = torch.randn(4, 5)
        symbols = torch.tensor([0, 1, 0, 1])
        out = model(x, symbols)
        assert out.shape == (4,)


def test_volatility_mlp_rejects_unknown_activation():
    import pytest

    with pytest.raises(ValueError):
        VolatilityMLP(n_numeric_features=5, n_symbols=2, activation="tanh")


def test_train_mlp_with_custom_loss_runs_and_predicts():
    rng = np.random.default_rng(0)
    n = 40
    X_train = rng.normal(size=(n, 3)).astype(np.float32)
    y_train = np.abs(rng.normal(size=n)).astype(np.float32) + 0.1
    symbols_train = rng.integers(0, 2, size=n)
    X_val, y_val, symbols_val = X_train[:10], y_train[:10], symbols_train[:10]

    model, scaler = train_mlp(
        X_train, symbols_train, y_train, X_val, symbols_val, y_val,
        n_symbols=2, epochs=3, batch_size=8, activation="gelu", loss_fn=rmspe_loss,
    )
    preds = mlp_predict(model, scaler, X_val, symbols_val)
    assert preds.shape == (10,)
    assert np.all(np.isfinite(preds))
