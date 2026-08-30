import numpy as np

from src.metrics import rmspe


def test_rmspe_zero_for_perfect_prediction():
    y = np.array([0.001, 0.002, 0.0005])
    assert rmspe(y, y) < 1e-6


def test_rmspe_hand_computed():
    y_true = np.array([1.0, 2.0])
    y_pred = np.array([1.1, 1.8])
    expected = np.sqrt(np.mean((((y_true - y_pred) / (y_true + 1e-6)) ** 2)))
    assert abs(rmspe(y_true, y_pred) - expected) < 1e-9
