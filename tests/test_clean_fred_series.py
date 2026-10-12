"""
Regression tests for the clean_fred_series frequency-inference data-loss bug.

Root cause (surfaced 2026-06-25 via scripts/stationarity_methods_check.py):
DataCleaner.clean_fred_series reindexed every FRED series with
    pd.date_range(start, end, freq=s.index.inferred_freq or "ME")
When a series' index is not perfectly regular, `inferred_freq` is None and the
code fell back to "ME" (month-END). For a MONTH-START series that lands the
reindex grid on 30th/31st dates that DON'T overlap the 1st-of-month data, so
reindex nulls EVERY value.

The real trigger (confirmed against live FRED data): UNRATE/CIVPART/CPI/Core CPI
publish with a ~1-month lag AND occasionally skip a month at the tail, leaving an
INTERNAL gap after dropna() (e.g. UNRATE returned ..Aug, Sep, [no Oct], Nov).
That gap makes the month-start index irregular -> inferred_freq None -> "ME"
fallback -> all-NaN. Those 4 macro features were therefore all-NaN in runs
#0/#1/#2 and silently dropped from the neural-net inputs. The same fallback also
downsampled holiday-irregular daily series (treasury yields, WTI, GVZ) to monthly.

Each reproduction test below asserts its fixture's precondition (inferred_freq is
None) so it cannot silently stop guarding the regression, then asserts the fix
preserves the data. They fail on the pre-fix code and pass with the fix.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data_cleaning.cleaner import DataCleaner


def _cleaner():
    # clean_fred_series uses no config state; an empty dict is enough.
    return DataCleaner({})


def test_month_start_series_with_internal_gap_survives():
    # Faithful to the real UNRATE/CIVPART failure: month-START dates with one
    # internal month missing (here: skip 2017-03-01). dropna leaves an irregular
    # index -> inferred_freq None -> pre-fix "ME" fallback nulled everything.
    idx = pd.date_range("2015-11-01", periods=24, freq="MS")
    idx = idx.delete(16)  # drop one interior month -> irregular
    assert idx.inferred_freq is None, "fixture precondition: must be irregular (None freq)"
    s = pd.Series(np.arange(len(idx), dtype=float), index=idx, name="U.S. Unemployment Rate")

    out = _cleaner().clean_fred_series(s)

    assert out.notna().all(), "month-start series must not be nulled by cleaning"
    # Every real observation is preserved at its original date.
    for ts, val in zip(idx, range(len(idx))):
        assert out.loc[ts] == float(val)


def test_daily_series_with_irregular_calendar_keeps_resolution():
    # Faithful to treasury yields / WTI / GVZ: business-daily with holiday gaps,
    # so the index is irregular and inferred_freq is None. Pre-fix the "ME"
    # fallback collapsed ~250 daily obs to ~12 monthly points.
    idx = pd.date_range("2020-01-01", periods=250, freq="B")
    idx = idx.delete([40, 95, 160])  # drop a few "holidays" -> irregular
    assert idx.inferred_freq is None, "fixture precondition: must be irregular (None freq)"
    s = pd.Series(np.linspace(1.0, 5.0, len(idx)), index=idx, name="U.S. Treasury Yields - 10Y")

    out = _cleaner().clean_fred_series(s)

    assert out.notna().all()
    # Distinct values stay high (daily), not collapsed to ~monthly.
    assert out.nunique() > 200, f"daily resolution lost: only {out.nunique()} distinct values"


def test_clean_regular_monthly_series_unchanged_by_fix():
    # A regular month-start series (inferred_freq "MS", e.g. UMCSENT) worked
    # pre-fix and must stay byte-identical post-fix -> the fix only changes the
    # previously-BROKEN series, which is what keeps runs #0/#1 comparable.
    idx = pd.date_range("2015-11-01", periods=24, freq="MS")
    assert idx.inferred_freq == "MS"
    s = pd.Series(np.linspace(60.0, 65.0, 24), index=idx, name="U.S. Consumer Sentiment")

    out = _cleaner().clean_fred_series(s)

    assert out.notna().all()
    assert out.nunique() == s.nunique()
    pd.testing.assert_series_equal(
        out.reset_index(drop=True), s.reset_index(drop=True), check_names=False
    )
