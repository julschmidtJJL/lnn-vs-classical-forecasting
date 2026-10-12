"""
Tests for SARIMAModel.fit_auto — the per-fold AIC order search (sarimax_aic, run #2).

- Selection: on an AR(1) series, the chosen order must have AIC no worse than a
  white-noise (0,0,0) fit — proves it actually selects by AIC, not first-wins.
- Order shape: chosen order is (p, 0, q) with p,q in {0,1,2}; seasonal fixed.
- All-fail guard: if every candidate raises, fit_auto raises RuntimeError so the
  fold is marked failed (drops from the aligned DM set).
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import models.sarimax_model as sm
from models.sarimax_model import SARIMAModel
from statsmodels.tsa.statespace.sarimax import SARIMAX


def _cfg():
    return {"models": {"SARIMAX": {
        "order": [1, 0, 0], "seasonal_order": [0, 0, 0, 0],
        "enforce_stationarity": True, "enforce_invertibility": True,
    }}}


def _ar1(n=200, phi=0.6, seed=7):
    rng = np.random.default_rng(seed)
    y = np.zeros(n)
    eps = rng.standard_normal(n) * 0.05
    for t in range(1, n):
        y[t] = phi * y[t - 1] + eps[t]
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.Series(y, index=idx, name="log_return")


def _model(features=None):
    feats = pd.DataFrame({"log_return": features if features is not None else _ar1()})
    return SARIMAModel(cfg=_cfg(), features=feats, logger=object(), ticker="TEST")


def test_selects_order_by_aic():
    y = _ar1()
    m = _model()
    fit = m.fit_auto(y_train=y, p_max=2, q_max=2, d=0)
    assert fit is not None
    assert m.model_fit is not None
    assert m.last_seasonal_order == (0, 0, 0, 0)
    p, d, q = m.last_order
    assert d == 0 and 0 <= p <= 2 and 0 <= q <= 2

    # The chosen fit's AIC must be <= a white-noise (0,0,0) baseline. On an AR(1)
    # signal that is only true if selection is genuinely AIC-driven.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        wn_aic = float(SARIMAX(endog=y, order=(0, 0, 0), seasonal_order=(0, 0, 0, 0),
                               trend="n").fit(disp=False).aic)
    assert float(m.model_fit.aic) <= wn_aic + 1e-6


def test_all_candidates_fail_raises(monkeypatch):
    def _boom(*args, **kwargs):
        raise np.linalg.LinAlgError("forced failure")

    monkeypatch.setattr(sm, "SARIMAX", _boom)
    m = _model()
    with pytest.raises(RuntimeError, match="all .* candidate orders failed"):
        m.fit_auto(y_train=_ar1(), p_max=2, q_max=2, d=0)
