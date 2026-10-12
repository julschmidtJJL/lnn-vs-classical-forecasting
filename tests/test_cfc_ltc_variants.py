"""
Tests for run #2 architecture/wiring variants in cfc_model.py / ltc_model.py.

Three concerns:
- REGRESSION: the default path (wiring="ncp", mode="default") must reproduce the
  #0/#1 build — AutoNCP sparse wiring, CfC default mode. Guards the "defaults
  preserve behaviour" claim in the run #2 plan.
- fc_backbone OUTPUT WIDTH: the CfC fc_backbone variant uses int units, so it
  MUST set proj_size=1 or the seq2one head (y_hat[:, -1, :].squeeze(-1)) leaves a
  [batch, units] tensor against a [batch] target. We assert the model's last-step
  output width is exactly 1 (out_features) — the bug Codex caught.
- LTC FullyConnected: builds + trains + produces finite predictions.
"""

import os
import sys

import numpy as np
import pandas as pd
import torch
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from models.cfc_model import CfCModel
from models.ltc_model import LTCModel
from ncps.wirings import AutoNCP, FullyConnected


class _FakeLogger:
    def log_result(self, **kwargs):
        return "fake-exp-id"

    def log_future_forecast(self, rows):
        pass

    def log_historic_forecast(self, row):
        pass


def _make_ar1_frame(n: int = 160, phi: float = 0.7, noise_sd: float = 0.05, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    y = np.zeros(n, dtype=np.float64)
    eps = rng.standard_normal(n) * noise_sd
    for t in range(1, n):
        y[t] = phi * y[t - 1] + eps[t]
    feat = np.roll(y, 1)
    feat[0] = 0.0
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.DataFrame({"log_return": y, "lag1": feat}, index=idx)


_BASE = {
    "units": 16, "lr": 0.01, "epochs": 2, "seq_len": 10, "clip_gradient": 1.0,
    "batch_first": True, "batch_size": 32, "sparsity_level": 0.5,
    "forecast_steps": 5, "scaler_kind": "zscore",
}


def _cfg(model_name):
    return {"models": {model_name: dict(_BASE)}}


def _last_step_width(model, n_features, seq_len=10):
    """Forward one window through the wrapped ncps cell; return the per-step
    output width that the seq2one head squeezes. Width 1 ⇒ squeeze(-1) → [batch]."""
    x = torch.zeros(1, seq_len, n_features, dtype=torch.float32)
    with torch.no_grad():
        y_hat, _ = model.forward(x)
    return y_hat[:, -1, :].shape[-1]


# ─── CfC ──────────────────────────────────────────────────────────────────────

class TestCfCVariants:

    def test_default_path_is_autoncp_default_mode(self):
        """REGRESSION: no params → AutoNCP wiring + default mode (the #0/#1 path)."""
        df = _make_ar1_frame()
        m = CfCModel(cfg=_cfg("CfC"), features=df, logger=_FakeLogger(), ticker="TEST")
        m.fit(df[["lag1"]].iloc[:120], df["log_return"].iloc[:120])
        assert m.last_params["wiring"] == "ncp"
        assert m.last_params["mode"] == "default"
        # AutoNCP wiring sets output_size=1, so the head width is 1.
        assert _last_step_width(m.model, n_features=1, seq_len=10) == 1

    def test_fc_backbone_output_width_is_one(self):
        """The fc_backbone trap: int units would emit `units` channels without
        proj_size. Assert the projected output width is exactly 1."""
        df = _make_ar1_frame()
        m = CfCModel(cfg=_cfg("CfC"), features=df, logger=_FakeLogger(), ticker="TEST")
        m.fit(df[["lag1"]].iloc[:120], df["log_return"].iloc[:120],
              params={"wiring": "fc_backbone", "mode": "default",
                      "backbone_units": 16, "backbone_layers": 1})
        assert m.last_params["wiring"] == "fc_backbone"
        assert m.last_params["backbone_units"] == 16
        width = _last_step_width(m.model, n_features=1, seq_len=10)
        assert width == 1, f"fc_backbone output width {width} != 1 — proj_size missing"

    @pytest.mark.parametrize("mode", ["default", "pure", "no_gate"])
    def test_modes_build_and_predict_finite(self, mode):
        df = _make_ar1_frame()
        m = CfCModel(cfg=_cfg("CfC"), features=df, logger=_FakeLogger(), ticker="TEST")
        X_tr, X_te = df[["lag1"]].iloc[:120], df[["lag1"]].iloc[120:]
        y_tr, y_te = df["log_return"].iloc[:120], df["log_return"].iloc[120:]
        fit = m.fit(X_tr, y_tr, params={"wiring": "ncp", "mode": mode})
        preds, _, _ = m.predict(model_fit=fit, X_train=X_tr, X_test=X_te, y_train=y_tr, y_test=y_te)
        assert m.last_params["mode"] == mode
        assert np.all(np.isfinite(preds.values))


# ─── LTC ──────────────────────────────────────────────────────────────────────

class TestLTCVariants:

    def test_default_path_is_autoncp(self):
        """REGRESSION: no params → AutoNCP wiring (the #0/#1 path)."""
        df = _make_ar1_frame()
        m = LTCModel(cfg=_cfg("LTC"), features=df, logger=_FakeLogger(), ticker="TEST")
        m.fit(df[["lag1"]].iloc[:120], df["log_return"].iloc[:120])
        assert m.last_params["wiring"] == "ncp"
        assert _last_step_width(m.model, n_features=1, seq_len=10) == 1

    def test_fully_connected_builds_and_predicts_finite(self):
        df = _make_ar1_frame()
        m = LTCModel(cfg=_cfg("LTC"), features=df, logger=_FakeLogger(), ticker="TEST")
        X_tr, X_te = df[["lag1"]].iloc[:120], df[["lag1"]].iloc[120:]
        y_tr, y_te = df["log_return"].iloc[:120], df["log_return"].iloc[120:]
        fit = m.fit(X_tr, y_tr, params={"wiring": "fully_connected"})
        preds, _, _ = m.predict(model_fit=fit, X_train=X_tr, X_test=X_te, y_train=y_tr, y_test=y_te)
        assert m.last_params["wiring"] == "fully_connected"
        assert _last_step_width(m.model, n_features=1, seq_len=10) == 1
        assert np.all(np.isfinite(preds.values))


# ─── mixed_memory (run #2 T3) ───────────────────────────────────────────────────

class TestMixedMemory:

    def test_default_records_mixed_memory_false(self):
        """REGRESSION: no mixed_memory param → meta records False (the #0/#1 path)."""
        df = _make_ar1_frame()
        c = CfCModel(cfg=_cfg("CfC"), features=df, logger=_FakeLogger(), ticker="TEST")
        c.fit(df[["lag1"]].iloc[:120], df["log_return"].iloc[:120])
        assert c.last_params["mixed_memory"] is False
        l = LTCModel(cfg=_cfg("LTC"), features=df, logger=_FakeLogger(), ticker="TEST")
        l.fit(df[["lag1"]].iloc[:120], df["log_return"].iloc[:120])
        assert l.last_params["mixed_memory"] is False

    @pytest.mark.parametrize("wiring,extra", [
        ("ncp", {}),
        ("fc_backbone", {"backbone_units": 16, "backbone_layers": 1}),
    ])
    def test_cfc_mixed_memory_builds_and_predicts_finite(self, wiring, extra):
        df = _make_ar1_frame()
        m = CfCModel(cfg=_cfg("CfC"), features=df, logger=_FakeLogger(), ticker="TEST")
        X_tr, X_te = df[["lag1"]].iloc[:120], df[["lag1"]].iloc[120:]
        y_tr, y_te = df["log_return"].iloc[:120], df["log_return"].iloc[120:]
        fit = m.fit(X_tr, y_tr, params={"wiring": wiring, "mode": "default",
                                        "mixed_memory": True, **extra})
        preds, _, _ = m.predict(model_fit=fit, X_train=X_tr, X_test=X_te, y_train=y_tr, y_test=y_te)
        assert m.last_params["mixed_memory"] is True
        assert _last_step_width(m.model, n_features=1, seq_len=10) == 1
        assert np.all(np.isfinite(preds.values))

    @pytest.mark.parametrize("raw,expected", [
        (False, False), (True, True),
        ("false", False), ("0", False), ("true", True), ("1", True),
    ])
    def test_mixed_memory_bool_coercion(self, raw, expected):
        """A stringy 'false'/'0' must NOT silently enable mixed_memory (bool('false') is True)."""
        df = _make_ar1_frame()
        m = CfCModel(cfg=_cfg("CfC"), features=df, logger=_FakeLogger(), ticker="TEST")
        m.fit(df[["lag1"]].iloc[:120], df["log_return"].iloc[:120],
              params={"wiring": "ncp", "mode": "default", "mixed_memory": raw})
        assert m.last_params["mixed_memory"] is expected

    def test_ltc_mixed_memory_builds_and_predicts_finite(self):
        df = _make_ar1_frame()
        m = LTCModel(cfg=_cfg("LTC"), features=df, logger=_FakeLogger(), ticker="TEST")
        X_tr, X_te = df[["lag1"]].iloc[:120], df[["lag1"]].iloc[120:]
        y_tr, y_te = df["log_return"].iloc[:120], df["log_return"].iloc[120:]
        fit = m.fit(X_tr, y_tr, params={"wiring": "ncp", "mixed_memory": True})
        preds, _, _ = m.predict(model_fit=fit, X_train=X_tr, X_test=X_te, y_train=y_tr, y_test=y_te)
        assert m.last_params["mixed_memory"] is True
        assert _last_step_width(m.model, n_features=1, seq_len=10) == 1
        assert np.all(np.isfinite(preds.values))
