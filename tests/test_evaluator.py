"""
Tests for src/evaluation/evaluator.py
Covers all 22 identified code paths in the Evaluator class and helper functions.
"""

import sys
import os

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.evaluation.evaluator import (
    Evaluator,
    apply_holm_bonferroni,
    directional_accuracy,
    dm_test_stat,
    evaluate_predictions,
    mae,
    rmse,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

RNG = np.random.default_rng(42)

# 24 folds × 21 steps = realistic walk-forward size
N_FOLDS = 24
HORIZON = 21


def _make_fold(n: int = HORIZON, seed: int = 0):
    """Returns (y_true, y_pred, residuals) as float arrays of length n."""
    rng = np.random.default_rng(seed)
    y_true = rng.standard_normal(n) * 0.01
    noise = rng.standard_normal(n) * 0.005
    y_pred = y_true + noise
    residuals = y_true - y_pred
    return y_true, y_pred, residuals


def _make_evaluator_with_folds(model_keys, n_folds=5) -> Evaluator:
    ev = Evaluator(horizon=HORIZON)
    for model_key in model_keys:
        for fold_id in range(n_folds):
            y_true, y_pred, residuals = _make_fold(seed=fold_id * 100)
            ev.add_fold(model_key, fold_id, y_true, y_pred, residuals)
    return ev


# ─── directional_accuracy ─────────────────────────────────────────────────────

class TestDirectionalAccuracy:

    def test_perfect_prediction(self):
        # All signs match: should be 1.0
        y_true = np.array([0.01, -0.02, 0.03])
        y_pred = np.array([0.005, -0.01, 0.02])
        assert directional_accuracy(y_true, y_pred) == pytest.approx(1.0)

    def test_all_wrong(self):
        y_true = np.array([0.01, -0.02])
        y_pred = np.array([-0.01, 0.02])
        assert directional_accuracy(y_true, y_pred) == pytest.approx(0.0)

    def test_half_correct(self):
        y_true = np.array([1, -1, 1, -1], dtype=float)
        y_pred = np.array([1, 1, -1, -1], dtype=float)
        assert directional_accuracy(y_true, y_pred) == pytest.approx(0.5)

    def test_zero_in_y_true_no_crash(self):
        # sign(0) == 0; predictors that also return 0 would match
        y_true = np.array([0.0, 0.01])
        y_pred = np.array([0.1, 0.005])
        result = directional_accuracy(y_true, y_pred)
        assert 0.0 <= result <= 1.0  # does not crash; value is valid

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="Shape mismatch"):
            directional_accuracy([1.0, 2.0], [1.0])


# ─── dm_test_stat ─────────────────────────────────────────────────────────────

class TestDmTestStat:

    def test_identical_forecasts_stat_near_zero(self):
        # Same errors → loss differential is zero → stat ≈ 0
        errors = RNG.standard_normal(HORIZON * N_FOLDS) * 0.01
        stat, p = dm_test_stat(errors, errors.copy(), h=1)
        # With d=0 everywhere, var_d_bar=0 → returns (0.0, 1.0)
        assert stat == pytest.approx(0.0, abs=1e-10)
        assert p == pytest.approx(1.0)

    def test_model1_clearly_better(self):
        # e1 = small errors, e2 = large errors → d = e1² - e2² < 0 → stat < 0 (e1 better)
        n = HORIZON * N_FOLDS
        e1 = RNG.standard_normal(n) * 0.001
        e2 = RNG.standard_normal(n) * 0.1
        stat, p = dm_test_stat(e1, e2, h=1)
        assert stat < 0  # positive stat means e2 better; negative means e1 better

    def test_minimum_fold_size_no_crash(self):
        # Minimum realistic series: one fold of 21 points
        e1 = RNG.standard_normal(HORIZON) * 0.01
        e2 = RNG.standard_normal(HORIZON) * 0.02
        stat, p = dm_test_stat(e1, e2, h=1)
        assert isinstance(stat, float)
        assert isinstance(p, float)
        assert 0.0 <= p <= 1.0

    def test_constant_differential_returns_finite(self):
        # Constant but different errors → d is constant and nonzero.
        # One model is always better, so the stat should be large in magnitude
        # (not NaN/inf). The function must not crash.
        e1 = np.ones(50) * 0.01
        e2 = np.ones(50) * 0.02
        stat, p = dm_test_stat(e1, e2, h=1)
        assert np.isfinite(stat)
        assert np.isfinite(p)
        assert 0.0 <= p <= 1.0


# ─── Evaluator.add_fold ───────────────────────────────────────────────────────

class TestEvaluatorAddFold:

    def test_single_fold_accumulates(self):
        ev = Evaluator(horizon=HORIZON)
        y_true, y_pred, res = _make_fold()
        ev.add_fold("sarimax", 0, y_true, y_pred, res)
        assert len(ev._folds["sarimax"]) == 1

    def test_multiple_folds_accumulate(self):
        ev = Evaluator(horizon=HORIZON)
        for i in range(5):
            y_true, y_pred, res = _make_fold(seed=i)
            ev.add_fold("lstm", i, y_true, y_pred, res)
        assert len(ev._folds["lstm"]) == 5

    def test_residual_length_mismatch_raises(self):
        ev = Evaluator(horizon=HORIZON)
        y_true, y_pred, _ = _make_fold()
        bad_residuals = np.ones(HORIZON + 3)  # wrong length
        with pytest.raises(ValueError, match="residuals length mismatch"):
            ev.add_fold("sarimax", 0, y_true, y_pred, residuals=bad_residuals)


# ─── Evaluator.compute ────────────────────────────────────────────────────────

class TestEvaluatorCompute:

    def test_two_models_five_folds(self):
        ev = _make_evaluator_with_folds(["sarimax", "lstm"], n_folds=5)
        results = ev.compute()
        assert "sarimax" in results
        assert "lstm" in results
        assert results["sarimax"]["rmse_mean"] > 0
        assert results["lstm"]["n_folds"] == 5

    def test_perfect_predictor_metrics(self):
        ev = Evaluator(horizon=HORIZON)
        y_true = RNG.standard_normal(HORIZON) * 0.01
        y_pred = y_true.copy()
        ev.add_fold("oracle", 0, y_true, y_pred)
        result = ev.compute()["oracle"]
        assert result["rmse_mean"] == pytest.approx(0.0, abs=1e-12)
        assert result["directional_accuracy_mean"] == pytest.approx(1.0)

    def test_no_folds_returns_empty(self):
        ev = Evaluator(horizon=HORIZON)
        assert ev.compute() == {}


# ─── Evaluator.get_significance_table ────────────────────────────────────────

class TestGetSignificanceTable:

    def test_two_models_one_row(self):
        ev = _make_evaluator_with_folds(["sarimax", "lstm"], n_folds=N_FOLDS)
        table = ev.get_significance_table()
        assert len(table) == 1
        assert set(table.columns) == {"model_a", "model_b", "dm_stat", "p_raw",
                                       "p_corrected", "significant"}

    def test_six_models_fifteen_rows(self):
        keys = ["sarimax", "varmax", "naive", "lstm", "ltc", "cfc"]
        ev = _make_evaluator_with_folds(keys, n_folds=N_FOLDS)
        table = ev.get_significance_table()
        assert len(table) == 15  # C(6, 2) = 15

    def test_zero_models_returns_empty_dataframe(self):
        ev = Evaluator(horizon=HORIZON)
        table = ev.get_significance_table()
        assert isinstance(table, pd.DataFrame)
        assert len(table) == 0

    def test_one_model_returns_empty_dataframe(self):
        ev = _make_evaluator_with_folds(["sarimax"], n_folds=5)
        table = ev.get_significance_table()
        assert len(table) == 0


# ─── apply_holm_bonferroni ────────────────────────────────────────────────────

class TestHolmBonferroni:

    def test_mixed_p_values(self):
        p_values = {"pair_a": 0.001, "pair_b": 0.04, "pair_c": 0.9}
        result = apply_holm_bonferroni(p_values, alpha=0.05)
        # After Holm correction the highly significant one should remain significant
        assert result["pair_a"]["significant"] is True
        assert result["pair_c"]["significant"] is False

    def test_all_large_p_values(self):
        p_values = {"a": 0.6, "b": 0.7, "c": 0.8}
        result = apply_holm_bonferroni(p_values, alpha=0.05)
        assert all(not v["significant"] for v in result.values())

    def test_empty_input(self):
        assert apply_holm_bonferroni({}) == {}


# ─── Evaluator.check_residuals ───────────────────────────────────────────────

class TestCheckResiduals:

    def test_heteroscedastic_residuals_detected(self):
        # ARCH-effected series: squared sine wave creates volatility clustering
        t = np.linspace(0, 4 * np.pi, 500)
        # Residuals with time-varying variance (ARCH structure)
        base = RNG.standard_normal(500)
        volatility = np.abs(np.sin(t)) + 0.1
        residuals = base * volatility

        ev = Evaluator(horizon=len(residuals))
        y_true = RNG.standard_normal(len(residuals)) * 0.01
        y_pred = y_true.copy()
        ev.add_fold("model", 0, y_true, y_pred, residuals=residuals)

        result = ev.check_residuals("model")
        assert "arch_stat" in result
        assert "arch_p" in result
        assert "ljung_box_p" in result
        # np.bool_ returned from statsmodels — use == not 'is'
        assert result["heteroscedastic"] == True  # noqa: E712

    def test_clean_residuals_not_flagged(self):
        # IID Gaussian residuals: no ARCH effect
        residuals = RNG.standard_normal(500) * 0.01
        ev = Evaluator(horizon=len(residuals))
        y_true = RNG.standard_normal(len(residuals)) * 0.01
        y_pred = y_true.copy()
        ev.add_fold("model", 0, y_true, y_pred, residuals=residuals)

        result = ev.check_residuals("model")
        assert result["heteroscedastic"] == False  # noqa: E712


# ─── evaluate_predictions (existing function: exception narrowing) ─────────────

class TestEvaluatePredictions:

    def test_metric_raises_type_error_gives_nan(self):
        def bad_metric(y_true, y_pred):
            raise TypeError("not numeric")

        result = evaluate_predictions([1.0, 2.0], [1.0, 2.0],
                                      metrics={"bad": bad_metric})
        assert np.isnan(result["bad"])

    def test_other_exceptions_propagate(self):
        # A RuntimeError should NOT be swallowed
        def very_bad_metric(y_true, y_pred):
            raise RuntimeError("something truly unexpected")

        with pytest.raises(RuntimeError):
            evaluate_predictions([1.0], [1.0],
                                  metrics={"bad": very_bad_metric})


# ─── check_residuals: non-finite guard (regression) ───────────────────────────
# Regression for 2026-06-08 smoke-run crash: a model that diverged to NaN
# predictions produced NaN residuals; het_arch(all_residuals) then raised
# MissingDataError ("exog contains inf or nans"), which escaped the caller's
# narrow `except ValueError` and crashed the whole walk-forward run BEFORE the
# experiment_summaries logging at main.py:987.

def test_check_residuals_skips_gracefully_on_all_nan_residuals():
    ev = Evaluator(horizon=HORIZON)
    for fold_id in range(5):
        y_true, y_pred, _ = _make_fold(seed=fold_id * 100)
        nan_res = np.full(HORIZON, np.nan)
        ev.add_fold("lstm", fold_id, y_true, y_pred, nan_res)

    # Must NOT raise; returns a graceful skipped-diagnostic dict.
    out = ev.check_residuals("lstm")
    assert np.isnan(out["arch_stat"])
    assert np.isnan(out["arch_p"])
    assert out["heteroscedastic"] is False
    assert "skipped" in out.get("note", "")


def test_check_residuals_drops_partial_nan_but_still_runs():
    ev = Evaluator(horizon=HORIZON)
    for fold_id in range(5):
        y_true, y_pred, residuals = _make_fold(seed=fold_id * 100)
        if fold_id == 0:
            residuals = residuals.copy()
            residuals[:3] = np.nan  # a few bad points, plenty finite remain
        ev.add_fold("ltc", fold_id, y_true, y_pred, residuals)

    out = ev.check_residuals("ltc")
    assert not np.isnan(out["arch_p"])          # diagnostic still computed
    assert "non-finite" in out.get("note", "")  # but the drop is recorded


def test_check_residuals_healthy_model_unchanged():
    ev = _make_evaluator_with_folds(["sarimax"], n_folds=5)
    out = ev.check_residuals("sarimax")
    assert not np.isnan(out["arch_p"])
    assert "note" not in out  # no drops on a clean model
