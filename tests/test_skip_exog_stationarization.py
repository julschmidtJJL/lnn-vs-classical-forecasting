"""
Regression test for the run #2b double-stationarization launch-blocker.

Found 2026-06-25 on the first real T5 smoke run: the #2b arm stationarized the
exog twice. The exog_stationarize table sets the 8 index columns to log_return
(applied by FeatureBuilder), then create_stationarity("transformed") ran
make_indices_data_stationary which log-diffs them AGAIN -> np.log(already-log-
returned) where values are negative/zero -> NaN/-inf -> every cfc/ltc/lstm fold
failed "Input X contains infinity".

Fix: create_stationarity(skip_exog_stationarization=True) skips make_indices +
make_macro (the table owns exog); internal price/financial transforms still run.
build_df_final passes the flag when cfg.experiment.exog_stationarize is set.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data_exploration.compliance import DataExplorationCompliance


def _features():
    idx = pd.date_range("2020-01-01", periods=60, freq="B")
    rng = np.random.default_rng(0)
    # gspc here stands in for an index column the exog table has ALREADY turned
    # into log-returns: small, with a negative and a zero so a second log() would
    # produce NaN/-inf (the double-transform we must avoid).
    gspc = rng.normal(0, 0.01, 60)
    gspc[5], gspc[9] = -0.02, 0.0
    return pd.DataFrame(
        {
            "open": np.linspace(100, 110, 60),
            "high": np.linspace(101, 111, 60),
            "low": np.linspace(99, 109, 60),
            "close": np.linspace(100, 110, 60),
            "gspc": gspc,
            "us_macro_x": np.linspace(1.0, 2.0, 60),
        },
        index=idx,
    )


def test_skip_exog_leaves_already_stationarized_exog_untouched_and_finite():
    comp = DataExplorationCompliance(_features())
    out = comp.create_stationarity("transformed", warmup_discard=5,
                                   skip_exog_stationarization=True)
    # The exog column is passed through unchanged (NOT re-log-diffed) -> finite.
    assert np.isfinite(out["gspc"].to_numpy()).all(), "exog must not be re-transformed -> no inf/NaN"
    # macro reporting-series derivatives are NOT appended (the table owns macro).
    assert not any(c.startswith("us_macro_x_") for c in out.columns)


def test_default_path_double_transforms_exog_reproducing_the_bug():
    # Guard the regression: WITHOUT the skip, make_indices_data_stationary log-diffs
    # the already-log-returned gspc -> log(negative/0) -> non-finite. This is the
    # exact pre-fix #2b failure the skip path prevents.
    comp = DataExplorationCompliance(_features())
    out = comp.create_stationarity("transformed", warmup_discard=5,
                                   skip_exog_stationarization=False)
    assert not np.isfinite(out["gspc"].to_numpy()).all(), "default path should reproduce the double-transform"
    # default path DOES append the macro reporting-series derivatives.
    assert any(c.startswith("us_macro_x_") for c in out.columns)
