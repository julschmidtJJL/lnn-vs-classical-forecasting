"""
Regression test for the LSTM NaN-loss divergence found in the 2026-06-08 cloud
smoke run. lstm.fit() fed X_train.values straight into StandardScaler without
sanitizing. The transformed condition carries all-NaN macro columns
(us_consumer_price_index, us_core_cpi, us_unemployment_rate,
us_labor_force_participation); StandardScaler ignores NaN when fitting but
passes it through on transform, so the sliding windows were NaN and the loss
was NaN from epoch 1 -> NaN predictions (RMSE=nan), which also crashed the
downstream residual diagnostics. ltc/cfc already sanitized in fit(); lstm did
not. Fix: lstm.fit() now calls sanitize_dataframe before scaling.

Pre-fix: model parameters become NaN. Post-fix: parameters stay finite and the
all-NaN column is dropped from feature_cols.
"""

import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from models.lstm_model import LSTMModel


def _xy(n=80, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "feat_ok": rng.standard_normal(n),
            "feat_partial_nan": rng.standard_normal(n),
            "feat_all_nan": np.full(n, np.nan),  # mirrors all-NaN macro columns
        }
    )
    X.loc[X.index[:5], "feat_partial_nan"] = np.nan  # stray NaN as well
    y = pd.Series(rng.standard_normal(n) * 0.01, name="log_return")
    return X, y


def test_lstm_fit_stays_finite_with_nan_features():
    torch.manual_seed(0)
    X, y = _xy()
    feats = X.copy()
    feats["log_return"] = y.values

    m = LSTMModel(cfg={}, features=feats, logger=None, ticker="TEST")
    params = {
        "hidden_size": 8,
        "num_layers": 1,
        "dropout": 0.0,
        "lr": 1e-3,
        "batch_size": 16,
        "epochs": 2,
        "seq_len": 5,
        "scaler_kind": "zscore",
    }
    m.fit(X_train=X, y_train=y, params=params, verbose=False)

    assert m.model is not None
    finite = all(torch.isfinite(p).all().item() for p in m.model.parameters())
    assert finite, "LSTM weights contain NaN/inf — NaN features leaked into training"

    # The all-NaN column must have been dropped by sanitize_dataframe.
    assert "feat_all_nan" not in m.feature_cols
    assert "feat_ok" in m.feature_cols
