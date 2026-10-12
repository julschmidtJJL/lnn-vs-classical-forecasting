"""
Tests for the run #2 SARIMAX fixed-order variant rows (ma1, arma11).

T1 of the run #2 build: the live dispatch (main.run_walk_forward_evaluation, the
`for run in runs` loop) must read a per-run order from `params.order` for variant
rows, while the bare `sarimax` row keeps its #0/#1 AR(1) identity. The order
resolution lives in the pure helper `main._resolve_sarimax_order`, tested here.

- ma1   -> (0, 0, 1)     pure MA(1)
- arma11 -> (1, 0, 1)    ARMA(1,1) mix
- bare sarimax (no params) -> config default (1, 0, 0)   [regression: identity preserved]
- config.yaml run-list actually carries the variant rows with distinct identities.
"""

import os
import sys

import pytest
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import main  # import-safe: secrets validated at logger construction, not import

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "src", "config.yaml")

# The fixed config default used by the bare `sarimax` (AR(1)) baseline.
_SARIMAX_CFG = {"order": [1, 0, 0], "seasonal_order": [0, 0, 0, 0]}


def test_ma1_variant_resolves_order_001():
    order, seas = main._resolve_sarimax_order(
        {"model": "sarimax", "variant": "ma1", "params": {"order": [0, 0, 1]}},
        _SARIMAX_CFG,
    )
    assert order == (0, 0, 1)
    assert seas == (0, 0, 0, 0)


def test_arma11_variant_resolves_order_101():
    order, seas = main._resolve_sarimax_order(
        {"model": "sarimax", "variant": "arma11", "params": {"order": [1, 0, 1]}},
        _SARIMAX_CFG,
    )
    assert order == (1, 0, 1)
    assert seas == (0, 0, 0, 0)


def test_bare_sarimax_keeps_config_default_order():
    # REGRESSION: a run with no params must fall back to the config AR(1) default,
    # so the bare `sarimax` run_key stays identical across #0/#1/#2.
    order, seas = main._resolve_sarimax_order({"model": "sarimax"}, _SARIMAX_CFG)
    assert order == (1, 0, 0)
    assert seas == (0, 0, 0, 0)


def test_empty_params_dict_falls_back_to_default():
    order, _ = main._resolve_sarimax_order(
        {"model": "sarimax", "params": {}}, _SARIMAX_CFG)
    assert order == (1, 0, 0)


def test_seasonal_order_override_respected():
    order, seas = main._resolve_sarimax_order(
        {"params": {"order": [0, 0, 1], "seasonal_order": [1, 0, 0, 5]}}, _SARIMAX_CFG)
    assert order == (0, 0, 1)
    assert seas == (1, 0, 0, 5)


def _runs():
    with open(_CONFIG_PATH) as fh:
        cfg = yaml.safe_load(fh)
    return cfg["experiment"]["runs"]


def test_config_carries_ma1_and_arma11_rows():
    runs = _runs()
    by_variant = {(r["model"], r.get("variant")): r for r in runs if r["model"] == "sarimax"}
    assert ("sarimax", "ma1") in by_variant
    assert ("sarimax", "arma11") in by_variant
    assert by_variant[("sarimax", "ma1")]["params"]["order"] == [0, 0, 1]
    assert by_variant[("sarimax", "arma11")]["params"]["order"] == [1, 0, 1]


def test_config_bare_sarimax_untouched():
    # The bare AR(1) row must carry no variant and no params override.
    runs = _runs()
    bare = [r for r in runs if r["model"] == "sarimax" and r.get("variant") is None]
    assert len(bare) == 1
    assert "params" not in bare[0] or not bare[0].get("params")


def test_sarimax_run_identities_are_distinct():
    # (model, variant) keys feed the run_key; they must be unique so ma1/arma11/bare
    # land as separate DM-comparable rows (no last-writer-wins collision).
    runs = _runs()
    ids = [(r["model"], r.get("variant")) for r in runs if r["model"] == "sarimax"]
    assert len(ids) == len(set(ids))
    assert ("sarimax", None) in ids  # bare AR(1) preserved


# ── Review fixes (Codex adversarial pass) ──────────────────────────────────────

def test_missing_config_order_falls_back_to_ar1_not_arima111():
    # FIX B: if models.SARIMAX.order is stripped, the bare baseline must stay AR(1),
    # never silently drift to integrated ARIMA(1,1,1).
    order, _ = main._resolve_sarimax_order({"model": "sarimax"}, {})
    assert order == (1, 0, 0)


def test_validate_rejects_present_but_empty_variant():
    # FIX C: a present-but-empty variant passes dedup yet collides with the bare
    # run_key — must be rejected at validation.
    cfg = {"experiment": {"runs": [
        {"model": "sarimax", "variant": "", "condition": "transformed"},
    ]}}
    with pytest.raises(ValueError, match="empty variant"):
        main.validate_experiment_config(cfg)


def test_non_finite_forecast_fold_is_dropped():
    # FIX A: a fold whose forecasts are NaN/inf must fail (drop), not be admitted
    # as a successful fold that poisons the DM tables.
    import numpy as np
    import pandas as pd

    n = 40
    y = pd.Series(np.linspace(0.0, 1.0, n), name="log_return")
    X = pd.DataFrame({"f": np.zeros(n)})
    # 4 folds; the splitter contract is (train_idx, test_idx) pairs.
    folds = [(np.arange(0, 10), np.arange(10, 20)),
             (np.arange(0, 20), np.arange(20, 30)),
             (np.arange(0, 25), np.arange(25, 30)),
             (np.arange(0, 30), np.arange(30, 40))]

    def fit_fn(ti):
        return object()

    def predict_fn(fit, ti, si):
        si = np.asarray(si)
        preds = np.zeros(len(si))
        if len(si) and si[0] == 20:          # fold index 1 -> emit a NaN forecast
            preds[0] = np.nan
        return preds, preds

    # success_threshold=0 so the model isn't excluded; we just check the bad fold drops.
    results = main._collect_model_folds("sarimax_ma1_transformed", folds, y, X,
                                        fit_fn=fit_fn, predict_fn=predict_fn,
                                        success_threshold=0.0)
    assert 0 in results and 2 in results and 3 in results
    assert 1 not in results  # the NaN fold was dropped, not admitted
