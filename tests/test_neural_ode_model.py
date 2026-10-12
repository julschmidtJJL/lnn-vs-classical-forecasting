"""
Tests for src/models/neural_ode_model.py — see test_ctrnn_model.py for the
same shape + sanity-learn pattern. Additional: solver-parity test asserts
both rk4 (fixed-step default) and dopri5 (adaptive) complete forward+backward
without error.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from models.neural_ode_model import NeuralODEModel, _NeuralODENetwork


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


def _make_sine_frame(n: int = 400, period: int = 20) -> pd.DataFrame:
    t = np.arange(n)
    y = np.sin(2.0 * np.pi * t / period)
    feat = np.roll(y, 1)
    feat[0] = 0.0
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.DataFrame({"log_return": y, "lag1": feat}, index=idx)


def _node(cfg_overrides=None, features=None):
    cfg = {
        "models": {
            "NeuralODE": {
                "latent_size": 16,
                "encoder_hidden": 16,
                "ode_hidden": 16,
                "solver": "rk4",
                "step_size": 0.25,
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
        cfg["models"]["NeuralODE"].update(cfg_overrides)
    return NeuralODEModel(cfg=cfg, features=features if features is not None else _make_sine_frame(),
                          logger=_FakeLogger(), ticker="TEST")


# ─── SHAPE TEST ───────────────────────────────────────────────────────────────────────────────────────────────────────

class TestNeuralODEShape:

    def test_predict_shape_and_no_nan(self):
        df = _make_sine_frame(n=150, period=20)
        m = _node(features=df, cfg_overrides={"epochs": 3, "seq_len": 10})
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
        assert np.all(np.isfinite(preds.values))


# ─── SOLVER PARITY ────────────────────────────────────────────────────────────────────────────────────────────────────

class TestNeuralODESolvers:
    """
    Per the eng review (D4-eng): both rk4 (default) and dopri5 (grid combo)
    must complete forward+backward on a tiny input. dopri5 is in scope as a
    smoke-validation comparison; this test guards against a bad odeint call.
    """

    @pytest.mark.parametrize("solver", ["rk4", "dopri5"])
    def test_solver_forward_backward(self, solver):
        torch.manual_seed(0)
        net = _NeuralODENetwork(
            n_features=2, latent_size=8, encoder_hidden=8, ode_hidden=8,
            solver=solver, step_size=0.25, rtol=1e-3, atol=1e-4,
        )
        x = torch.randn(4, 10, 2)
        y_target = torch.randn(4)
        out = net(x)
        assert out.shape == (4,)
        assert torch.all(torch.isfinite(out))
        loss = ((out - y_target) ** 2).mean()
        loss.backward()
        # Confirm gradients flowed to encoder, ode_func, and decoder.
        assert net.encoder.weight_ih_l0.grad is not None
        assert net.ode_func.net[0].weight.grad is not None
        assert net.decoder.weight.grad is not None


# ─── SANITY LEARN TEST ────────────────────────────────────────────────────────────────────────────────────────────────

class TestNeuralODELearns:

    def test_learns_sine_signal(self):
        torch.manual_seed(42)
        np.random.seed(42)

        df = _make_sine_frame(n=400, period=20)
        m = _node(features=df, cfg_overrides={
            "latent_size": 16, "encoder_hidden": 16, "ode_hidden": 16,
            "epochs": 200, "lr": 0.01, "seq_len": 20, "batch_size": 64, "solver": "rk4",
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

        assert signal_std > 0
        ratio = rmse / signal_std
        assert ratio < 0.5, (
            f"Neural ODE failed to learn sine signal: RMSE/std = {ratio:.3f} "
            f"(rmse={rmse:.4f}, signal_std={signal_std:.4f})."
        )
