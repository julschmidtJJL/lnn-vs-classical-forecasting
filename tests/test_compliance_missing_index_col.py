"""
Regression test for the KeyError crash on the SPY full-grid run.

Root cause (Hetzner run 2026-06-09T08:11:29Z): compliance.make_indices_data_stationary
iterated over a hardcoded list of index columns
    ["gspc", "ndx", "dji", "gdaxi", "stoxx50e", "gld", "urth", "acwi"]
and unconditionally did np.log(df_new[c]).diff(). But the per-ticker leakage
guard in FeatureBuilder legitimately drops "^GSPC" (sanitized "gspc") when
forecasting SPY (feature_exclusions: {SPY: ["^GSPC"]}), because the S&P 500
index is a near-direct proxy of the SPY target. With "gspc" absent the loop
raised:
    KeyError: 'gspc'   (compliance.py:439)
This killed SPY on every retry of the authoritative grid (AAPL had already
logged; JPM never ran because it was queued behind SPY).

The fix guards the loop with `if c in df_new.columns`. These tests fail on the
pre-fix code (KeyError) and pass with the guard in place.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data_exploration.compliance import DataExplorationCompliance


def test_missing_gspc_column_does_not_raise():
    # SPY case: leakage guard already dropped "gspc"; other index cols remain.
    idx = pd.date_range("2020-01-01", periods=5, freq="B")
    df = pd.DataFrame(
        {
            "ndx": [100.0, 101.0, 102.0, 103.0, 104.0],
            "dji": [200.0, 201.0, 202.0, 203.0, 204.0],
            "gld": [50.0, 51.0, 52.0, 53.0, 54.0],
        },
        index=idx,
    )
    comp = DataExplorationCompliance(df)
    out = comp.make_indices_data_stationary(df)  # must not raise KeyError

    assert "gspc" not in out.columns
    # Present columns are still log-differenced (first row NaN from .diff()).
    expected = np.log(df["ndx"]).diff()
    pd.testing.assert_series_equal(out["ndx"], expected, check_names=False)


def test_present_index_columns_unaffected_by_guard():
    # When every index column is present, behaviour is unchanged.
    idx = pd.date_range("2020-01-01", periods=4, freq="B")
    cols = ["gspc", "ndx", "dji", "gdaxi", "stoxx50e", "gld", "urth", "acwi"]
    df = pd.DataFrame(
        {c: np.linspace(100.0, 110.0, len(idx)) for c in cols}, index=idx
    )
    comp = DataExplorationCompliance(df)
    out = comp.make_indices_data_stationary(df)

    for c in cols:
        pd.testing.assert_series_equal(
            out[c], np.log(df[c]).diff(), check_names=False
        )
