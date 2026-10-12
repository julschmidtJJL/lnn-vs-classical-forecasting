"""
Tests for src/models/ctrnn_model.py

Two layers:
- Shape test: instantiate, fit, predict on tiny synthetic frame; asserts the
  contract that the walk-forward harness depends on (shapes + no NaNs).
- Sanity-learn test: seeded AR(1) signal; assert RMSE/signal_std < 0.3 on
  held-out window. Catches dead gradients, scaler-axis swaps, batch-dim bugs.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from models.ctrnn_model import CTRNNModel


class _FakeLogger:
    def __init__(self):
        self.results = []

    def log_result(self, **kwargs):
        self.results.append(kwargs)
        return "fake-exp-id"

    def log_future_forecast(self, rows):
        pass

    def log_historic_forecast(self, row):
        pass


def _make_ar1_frame(n: int = 400, phi: float = 0.7, noise_sd: float = 0.05, seed: int = 42) -> pd.DataFrame:
    """Small AR(1) frame for shape tests only — see _make_sine_frame for the learn test."""
    rng = np.random.default_rng(seed)
    y = np.zeros(n, dtype=np.float64)
    eps = rng.standard_normal(n) * noise_sd
    for t in range(1, n):
        y[t] = phi * y[t - 1] + eps[t]
    feat = np.roll(y, 1)
    feat[0] = 0.0
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.DataFrame({"log_return": y, "lag1": feat}, index=idx)


def _make_sine_frame(n: int = 400, period: int = 20) -> pd.DataFrame:
    """
    Clean sine wave with one lagged feature. Perfect predictability (no noise)
    gives a sharp sanity floor near zero. Any working model should hit RMSE/std
    well below 0.5 within a few hundred epochs.
    """
    t = np.arange(n)
    y = np.sin(2.0 * np.pi * t / period)
    feat = np.roll(y, 1)
    feat[0] = 0.0
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.DataFrame({"log_return": y, "lag1": feat}, index=idx)


def _ctrnn(cfg_overrides=None, features=None):
    cfg = {
        "models": {
            "CTRNN": {
                "hidden_size": 16,
                "dt": 1.0,
                "lr": 0.01,
                "batch_size": 32,
                "epochs": 5,
                "seq_len": 10,
                "grad_clip": 1.0,
                "forecast_steps": 5,
                "scaler_kind": "zscore",
            }
        }
    }
    if cfg_overrides:
        cfg["models"]["CTRNN"].update(cfg_overrides)
    return CTRNNModel(cfg=cfg, features=features if features is not None else _make_ar1_frame(),
                      logger=_FakeLogger(), ticker="TEST")


# ─── SHAPE TEST ───────────────────────────────────────────────────────────────────────────────────────────────────────

class TestCTRNNShape:

    def test_fit_returns_module(self):
        df = _make_ar1_frame(n=120)
        m = _ctrnn(features=df, cfg_overrides={"epochs": 2, "seq_len": 8})
        X_train = df[["lag1"]].iloc[:80]
        y_train = df["log_return"].iloc[:80]
        out = m.fit(X_train, y_train)
        assert out is not None
        assert m.model is not None
        assert m.scaler_X is not None and m.scaler_y is not None

    def test_predict_shape_and_no_nan(self):
        df = _make_ar1_frame(n=150)
        m = _ctrnn(features=df, cfg_overrides={"epochs": 3, "seq_len": 10})
        X_train = df[["lag1"]].iloc[:100]
        X_test = df[["lag1"]].iloc[100:]
        y_train = df["log_return"].iloc[:100]
        y_test = df["log_return"].iloc[100:]

        fit = m.fit(X_train, y_train)
        preds, resids, exp_id = m.predict(
            model_fit=fit, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test
        )

        assert len(preds) == len(y_test)
        assert len(resids) == len(y_test)
        # Per-fold logging is centralized in main._log_folds_centrally;
        # predict() must not log and returns exp_id=None in the legacy slot.
        assert exp_id is None
        assert m.logger.results == []
        # First prediction should be finite (window includes train history)
        assert np.isfinite(preds.iloc[0])
        # All-test-window predictions should be finite
        assert np.all(np.isfinite(preds.values))


# ─── SANITY LEARN TEST ────────────────────────────────────────────────────────────────────────────────────────────────

class TestCTRNNLearns:

    def test_learns_sine_signal(self):
        """
        Seeded sine-wave signal with a lag-1 feature. Perfect predictability →
        any working model should hit RMSE/signal_std well below 0.5 in 200
        epochs. Catches dead gradients, scaler-axis swaps, batch-dim bugs,
        and tau-init pathologies.
        """
        torch.manual_seed(42)
        np.random.seed(42)

        df = _make_sine_frame(n=400, period=20)
        m = _ctrnn(features=df, cfg_overrides={
            "hidden_size": 32, "epochs": 200, "lr": 0.01, "seq_len": 20, "batch_size": 64,
        })
        cut = 320
        X_train = df[["lag1"]].iloc[:cut]
        X_test = df[["lag1"]].iloc[cut:]
        y_train = df["log_return"].iloc[:cut]
        y_test = df["log_return"].iloc[cut:]

        fit = m.fit(X_train, y_train)
        preds, _, _ = m.predict(model_fit=fit, X_train=X_train, X_test=X_test,
                                y_train=y_train, y_test=y_test)

        yp = preds.to_numpy(dtype=float)
        yt = y_test.to_numpy(dtype=float)
        signal_std = float(np.std(yt))
        rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))

        assert signal_std > 0, "test signal has zero variance; bad fixture"
        ratio = rmse / signal_std
        assert ratio < 0.5, (
            f"CT-RNN failed to learn sine signal: RMSE/std = {ratio:.3f} "
            f"(rmse={rmse:.4f}, signal_std={signal_std:.4f}). "
            "Check gradients, scaler, tau init."
        )
