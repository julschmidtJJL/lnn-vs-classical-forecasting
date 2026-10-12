"""
Tests for Evaluator.friedman_cross_asset (added by T5 per the codex review).
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from evaluation.evaluator import Evaluator


def _ev():
    return Evaluator(horizon=21)


class TestFriedmanCrossAsset:

    def test_consistent_ranking_across_assets(self):
        """
        Same model order is best on every asset. Should produce a small (but
        finite) p-value reflecting consistent ranks. With 3 assets and 5 models
        and identical rank vectors [1,2,3,4,5] each, Friedman χ² = n*(k-1) = 12,
        df=4 → p ≈ 0.0174.
        """
        ev = _ev()
        rmses = {
            "AAPL": {"naive": 0.5, "lstm": 0.4, "ltc": 0.3, "ctrnn": 0.2, "neural_ode": 0.1},
            "SPY":  {"naive": 5.0, "lstm": 4.0, "ltc": 3.0, "ctrnn": 2.0, "neural_ode": 1.0},
            "JPM":  {"naive": 50.0, "lstm": 40.0, "ltc": 30.0, "ctrnn": 20.0, "neural_ode": 10.0},
        }
        out = ev.friedman_cross_asset(rmses)

        assert out["n_assets"] == 3
        assert out["n_models"] == 5
        assert set(out["models"]) == {"naive", "lstm", "ltc", "ctrnn", "neural_ode"}
        # Identical rank order per asset → small p
        assert out["p_value"] < 0.05
        # Each asset's ranks should be a permutation of [1..5]
        for a in out["assets"]:
            assert sorted(out["ranks_by_asset"][a]) == [1.0, 2.0, 3.0, 4.0, 5.0]
        # Sanity: descriptive caveat present
        assert "descriptive" in out["note"]

    def test_random_ranking_high_p(self):
        """No ranking pattern across assets → large p."""
        ev = _ev()
        # Construct rankings that are scrambled per asset
        rmses = {
            "A": {"m1": 1.0, "m2": 2.0, "m3": 3.0},
            "B": {"m1": 3.0, "m2": 1.0, "m3": 2.0},
            "C": {"m1": 2.0, "m2": 3.0, "m3": 1.0},
        }
        out = ev.friedman_cross_asset(rmses)
        # Average rank = 2 for all → χ² = 0 → p = 1
        assert out["p_value"] > 0.5

    def test_model_filter(self):
        """Explicit models arg restricts the comparison."""
        ev = _ev()
        rmses = {
            "A": {"m1": 0.1, "m2": 0.2, "m3": 0.3, "m4": 0.4},
            "B": {"m1": 0.1, "m2": 0.2, "m3": 0.3, "m4": 0.4},
        }
        out = ev.friedman_cross_asset(rmses, models=["m1", "m2", "m3"])
        assert out["models"] == ["m1", "m2", "m3"]
        assert out["n_models"] == 3

    def test_ragged_input_raises(self):
        ev = _ev()
        rmses = {
            "A": {"m1": 0.1, "m2": 0.2, "m3": 0.3},
            "B": {"m1": 0.1, "m2": 0.2},  # missing m3
        }
        with pytest.raises(ValueError, match="Missing RMSE"):
            ev.friedman_cross_asset(rmses, models=["m1", "m2", "m3"])

    def test_too_few_models_raises(self):
        ev = _ev()
        rmses = {"A": {"m1": 0.1, "m2": 0.2}, "B": {"m1": 0.1, "m2": 0.2}}
        with pytest.raises(ValueError, match="≥ 3 treatments"):
            ev.friedman_cross_asset(rmses)

    def test_too_few_assets_raises(self):
        ev = _ev()
        rmses = {"A": {"m1": 0.1, "m2": 0.2, "m3": 0.3}}
        with pytest.raises(ValueError, match="≥ 2 assets"):
            ev.friedman_cross_asset(rmses)
