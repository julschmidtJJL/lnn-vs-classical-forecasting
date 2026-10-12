"""
Tests for src/models/naive_model.py
"""

import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from models.naive_model import NaiveBaseline


def _make_baseline():
    """Returns a NaiveBaseline with minimal dummy config."""
    return NaiveBaseline(cfg={}, features=None, logger=None, ticker="AAPL")


class TestNaiveBaselinePredictFold:

    def test_first_prediction_equals_last_training_value(self):
        nb = _make_baseline()
        y_train = np.array([0.01, -0.02, 0.03, 0.005])
        y_test = np.array([0.02, -0.01, 0.004])
        preds, _ = nb.predict_fold(y_train, y_test)
        assert preds[0] == pytest.approx(y_train[-1])  # 0.005

    def test_subsequent_predictions_equal_lagged_actuals(self):
        nb = _make_baseline()
        y_train = np.array([0.01, 0.02])
        y_test = np.array([0.03, 0.04, 0.05])
        preds, _ = nb.predict_fold(y_train, y_test)
        # pred[1] = y_test[0], pred[2] = y_test[1]
        assert preds[1] == pytest.approx(y_test[0])
        assert preds[2] == pytest.approx(y_test[1])

    def test_residuals_equal_y_test_minus_predictions(self):
        nb = _make_baseline()
        y_train = np.array([0.01])
        y_test = np.array([0.02, -0.01, 0.03])
        preds, resids = nb.predict_fold(y_train, y_test)
        expected_resids = y_test - preds
        np.testing.assert_allclose(resids, expected_resids)

    def test_single_step_horizon(self):
        # n=1: only pred[0] = y_train[-1]
        nb = _make_baseline()
        y_train = np.array([0.01, 0.02, 0.03])
        y_test = np.array([0.04])
        preds, resids = nb.predict_fold(y_train, y_test)
        assert len(preds) == 1
        assert preds[0] == pytest.approx(y_train[-1])
        assert resids[0] == pytest.approx(y_test[0] - y_train[-1])

    def test_output_shapes_match_y_test(self):
        nb = _make_baseline()
        n_test = 21
        y_train = np.random.default_rng(0).standard_normal(750) * 0.01
        y_test = np.random.default_rng(1).standard_normal(n_test) * 0.01
        preds, resids = nb.predict_fold(y_train, y_test)
        assert len(preds) == n_test
        assert len(resids) == n_test

    def test_accepts_pandas_series_input(self):
        import pandas as pd
        nb = _make_baseline()
        y_train = pd.Series([0.01, -0.02, 0.03])
        y_test = pd.Series([0.02, -0.01])
        preds, resids = nb.predict_fold(y_train, y_test)
        assert len(preds) == 2
        assert preds[0] == pytest.approx(0.03)

    def test_empty_y_train_raises(self):
        nb = _make_baseline()
        with pytest.raises(ValueError, match="y_train must be non-empty"):
            nb.predict_fold(np.array([]), np.array([0.01]))

    def test_empty_y_test_raises(self):
        nb = _make_baseline()
        with pytest.raises(ValueError, match="y_test must be non-empty"):
            nb.predict_fold(np.array([0.01]), np.array([]))

    def test_realistic_walk_forward_fold(self):
        # Simulate a realistic fold: 750 training points, 21 test points
        rng = np.random.default_rng(42)
        y_train = rng.standard_normal(750) * 0.01
        y_test = rng.standard_normal(21) * 0.01
        preds, resids = _make_baseline().predict_fold(y_train, y_test)
        assert len(preds) == 21
        assert len(resids) == 21
        assert preds[0] == pytest.approx(y_train[-1])
        np.testing.assert_allclose(preds[1:], y_test[:-1])
