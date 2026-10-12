"""
Run #2 config-validation tests: variant-aware dedup + sarimax_aic registration.

validate_experiment_config is a pure, data-free guard (main.py), so these run
fast and lock the run-list contract the variant grid depends on.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from main import validate_experiment_config


def _cfg(runs):
    return {"experiment": {"runs": runs}}


def test_same_model_different_variant_is_allowed():
    """Six CfC variants share model='cfc' but differ by variant id — all valid."""
    runs = [
        {"model": "cfc", "variant": "default_ncp", "condition": "transformed"},
        {"model": "cfc", "variant": "pure_ncp", "condition": "transformed"},
        {"model": "cfc", "variant": "default_fc", "condition": "transformed"},
    ]
    validate_experiment_config(_cfg(runs))  # no raise


def test_duplicate_variant_is_rejected():
    runs = [
        {"model": "cfc", "variant": "pure_ncp", "condition": "transformed"},
        {"model": "cfc", "variant": "pure_ncp", "condition": "transformed"},
    ]
    with pytest.raises(ValueError, match="duplicate entry"):
        validate_experiment_config(_cfg(runs))


def test_sarimax_aic_is_valid_classical_transformed():
    validate_experiment_config(_cfg([{"model": "sarimax_aic", "condition": "transformed"}]))


def test_sarimax_aic_rejected_on_raw():
    with pytest.raises(ValueError, match="classical model 'sarimax_aic' cannot run on condition 'raw'"):
        validate_experiment_config(_cfg([{"model": "sarimax_aic", "condition": "raw"}]))


def test_non_variant_runs_still_dedup_by_model_condition():
    """Regression: the #0/#1 (model, condition) dedup still fires when no variant."""
    runs = [
        {"model": "lstm", "condition": "transformed"},
        {"model": "lstm", "condition": "transformed"},
    ]
    with pytest.raises(ValueError, match="duplicate entry"):
        validate_experiment_config(_cfg(runs))
