"""
Regression test for the inf leak in quarterly-fundamental stationarization.

Root cause (investigate skill, 2026-06-08): compliance.py:stationarize_reporting_series
computed reports.pct_change() with no zero-denominator guard. When a prior
quarterly report value is 0.0, pct_change yields +/-inf (undefined "0 -> X"
growth). The trailing fillna(0.0) only sanitized NaN, so inf leaked into
{col}_pctchg and {col}_pctchg_ffill, and the sklearn scaler inside the
lstm/ltc/cfc walk-forward fits raised:
    "Input X contains infinity or a value too large for dtype('float64')."
This killed every neural fold on the transformed condition (real data: AAPL
income_q_depreciation, cashflow_q_*_long_term_debt_*, balance_sheet_q_short_long_term_debt_total).

These tests fail on the pre-fix code (inf present) and pass with the
.replace([inf, -inf], NaN) guard in place.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data_exploration.compliance import DataExplorationCompliance


def _quarterly_frame(values):
    """Build a daily-frequency frame with one quarterly fundamental column
    that is forward-filled between sparse report dates (mirrors how SEC
    fundamentals enter the pipeline as income_q_* columns)."""
    # Four quarterly report dates; value held flat until the next report.
    report_dates = pd.to_datetime(
        ["2021-09-27", "2021-12-27", "2022-03-28", "2022-06-27"]
    )
    idx = pd.date_range("2021-09-27", "2022-09-01", freq="B")
    s = pd.Series(np.nan, index=idx, dtype=float)
    for d, v in zip(report_dates, values):
        s.loc[d] = v
    s = s.ffill()
    return pd.DataFrame({"income_q_depreciation": s})


def test_zero_to_nonzero_report_does_not_leak_inf():
    # Prior report 0.0 then a positive value -> pct_change divides by zero.
    df = _quarterly_frame([0.0, 0.0, 1234.0, 1500.0])
    comp = DataExplorationCompliance(df)
    out = comp.stationarize_reporting_series(df, prefixes=("income_q",))

    for col in ("income_q_depreciation_pctchg", "income_q_depreciation_pctchg_ffill"):
        assert col in out.columns, f"missing expected column {col}"
        vals = out[col].to_numpy()
        assert not np.isinf(vals).any(), f"{col} still contains inf after fix"
        assert not np.isnan(vals).any(), f"{col} should be NaN-free (fillna(0.0))"


def test_normal_reports_unaffected_by_guard():
    # No zero denominators: guard must not alter legitimate pct-change values.
    df = _quarterly_frame([100.0, 110.0, 121.0, 133.1])
    comp = DataExplorationCompliance(df)
    out = comp.stationarize_reporting_series(df, prefixes=("income_q",))

    pct = out["income_q_depreciation_pctchg"]
    # 10% step between consecutive distinct reports (pandas pct_change is a ratio).
    nonzero = pct[pct != 0.0]
    assert len(nonzero) >= 2
    assert np.allclose(nonzero.to_numpy(), 0.10, atol=1e-9), nonzero.to_numpy()
