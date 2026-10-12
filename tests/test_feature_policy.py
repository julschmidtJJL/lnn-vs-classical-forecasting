"""
Tests for the past-target feature policy (run #0 vs run #1).

Two layers:
- Unit: models.feature_policy.include_past_target / apply_target_policy behave
  correctly for the on/off flag, missing keys, and missing column.
- Integration: a real (tiny) CTRNN fit honours the flag end-to-end — the
  recorded feature_cols include 'log_return' ONLY when the experiment opts in,
  and the fitted model still produces finite 1-step-ahead predictions. This is
  the guard against the "double-drop" footgun: log_return is removed both in
  main.py and inside each model.fit, so both layers must read the same flag.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from models.feature_policy import apply_target_policy, include_past_target, TARGET_COL
from models.ctrnn_model import CTRNNModel
from models.lstm_model import LSTMModel


# ── Unit: include_past_target ──────────────────────────────────────────────────

def test_include_default_false():
    assert include_past_target({}) is False
    assert include_past_target({"experiment": {}}) is False


def test_include_true_when_flagged():
    cfg = {"experiment": {"include_past_target_as_feature": True}}
    assert include_past_target(cfg) is True


def test_include_false_when_flagged_off():
    cfg = {"experiment": {"include_past_target_as_feature": False}}
    assert include_past_target(cfg) is False


def test_include_tolerates_none_and_garbage():
    assert include_past_target(None) is False
    assert include_past_target("not-a-dict") is False


# ── Unit: apply_target_policy ──────────────────────────────────────────────────

def _frame():
    return pd.DataFrame({TARGET_COL: [0.1, 0.2, 0.3], "feat_a": [1.0, 2.0, 3.0]})


def test_policy_drops_target_when_off():
    out = apply_target_policy(_frame(), {})
    assert TARGET_COL not in out.columns
    assert "feat_a" in out.columns


def test_policy_keeps_target_when_on():
    cfg = {"experiment": {"include_past_target_as_feature": True}}
    out = apply_target_policy(_frame(), cfg)
    assert TARGET_COL in out.columns
    assert "feat_a" in out.columns


def test_policy_noop_when_target_absent():
    df = pd.DataFrame({"feat_a": [1.0, 2.0]})
    out = apply_target_policy(df, {})
    assert list(out.columns) == ["feat_a"]


def test_policy_passthrough_none():
    assert apply_target_policy(None, {}) is None


# ── Integration: CTRNN fit honours the flag end-to-end ─────────────────────────

def _ar1_frame(n: int = 240, phi: float = 0.6, seed: int = 7) -> pd.DataFrame:
    """log_return + two exogenous features so X is non-empty with the target dropped."""
    rng = np.random.default_rng(seed)
    y = np.zeros(n)
    eps = rng.standard_normal(n) * 0.05
    for t in range(1, n):
        y[t] = phi * y[t - 1] + eps[t]
    lag1 = np.roll(y, 1); lag1[0] = 0.0
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.DataFrame({"log_return": y, "lag1": lag1, "noise": rng.standard_normal(n)}, index=idx)


def _ctrnn(include_target: bool, features: pd.DataFrame) -> CTRNNModel:
    cfg = {
        "experiment": {"include_past_target_as_feature": include_target},
        "models": {"CTRNN": {
            "hidden_size": 8, "dt": 1.0, "lr": 0.01, "batch_size": 32,
            "epochs": 3, "seq_len": 10, "grad_clip": 1.0, "forecast_steps": 5,
            "scaler_kind": "zscore",
        }},
    }
    return CTRNNModel(cfg=cfg, features=features, logger=None, ticker="TEST")


def test_fit_excludes_target_when_off():
    df = _ar1_frame()
    X, y = df, df["log_return"]
    m = _ctrnn(include_target=False, features=df)
    m.fit(X_train=X, y_train=y)
    assert TARGET_COL not in m.feature_cols
    assert set(m.feature_cols) == {"lag1", "noise"}


def test_fit_includes_target_when_on():
    df = _ar1_frame()
    X, y = df, df["log_return"]
    m = _ctrnn(include_target=True, features=df)
    m.fit(X_train=X, y_train=y)
    assert TARGET_COL in m.feature_cols
    assert set(m.feature_cols) == {"log_return", "lag1", "noise"}


def test_train_test_split_honours_flag():
    """The legacy LSTM/LTC/CfC train_test_split() splitter must obey the same
    flag as fit()/main.py — otherwise a caller using it silently gets the run #0
    feature set despite opting in (the third-gate footgun Codex flagged)."""
    df = _ar1_frame()
    off = LSTMModel(cfg={"experiment": {"include_past_target_as_feature": False}},
                    features=df, logger=None, ticker="T").train_test_split()
    on = LSTMModel(cfg={"experiment": {"include_past_target_as_feature": True}},
                   features=df, logger=None, ticker="T").train_test_split()
    assert TARGET_COL not in off[0].columns          # X_train, flag off
    assert TARGET_COL in on[0].columns               # X_train, flag on


def test_fit_predict_finite_with_target_feature():
    """With the autoregressive feature on, fit->predict still yields finite,
    correctly-shaped 1-step-ahead predictions (no NaN/shape regressions)."""
    df = _ar1_frame()
    split = 180
    X_tr, X_te = df.iloc[:split], df.iloc[split:]
    y_tr, y_te = df["log_return"].iloc[:split], df["log_return"].iloc[split:]
    m = _ctrnn(include_target=True, features=df)
    fit = m.fit(X_train=X_tr, y_train=y_tr)
    preds, resids, _ = m.predict(model_fit=fit, X_train=X_tr, X_test=X_te,
                                 y_train=y_tr, y_test=y_te)
    preds = np.asarray(preds, dtype=float)
    assert preds.shape[0] == len(y_te)
    assert np.isfinite(preds).all()
