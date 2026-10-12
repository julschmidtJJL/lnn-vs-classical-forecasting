"""
Tests for the feature-matrix data-quality gate (src/validation/data_quality.py).

The gate exists to catch SILENT data-degradation: a column that builds without
raising but carries no usable information. These tests pin the severity split
that matters — all-NaN and daily-collapse are hard ERRORS; zero-variance and
high-NaN are non-fatal WARNINGS — plus the ALLOW_DATA_QUALITY_FAIL override.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from validation.data_quality import (
    DataQualityError,
    QualityIssue,
    validate_feature_matrix,
)


def _clean_matrix(n=120):
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "log_return": rng.normal(size=n),
            "us_treasury_yields_10y": np.linspace(1.0, 3.0, n) + rng.normal(0, 0.01, n),
        },
        index=idx,
    )


def test_clean_matrix_passes_with_no_issues():
    issues = validate_feature_matrix(_clean_matrix(), context="clean")
    assert issues == []


def test_all_nan_column_is_a_hard_error():
    df = _clean_matrix()
    df["us_unemployment_rate"] = np.nan  # the exact bug we hit
    with pytest.raises(DataQualityError) as ei:
        validate_feature_matrix(df, context="AAPL/raw")
    codes = {i.code for i in ei.value.issues}
    assert "all_nan" in codes
    assert any(i.column == "us_unemployment_rate" for i in ei.value.issues)


def test_daily_series_collapsed_to_monthly_is_an_error():
    df = _clean_matrix(n=250)
    # Simulate the downsampling bug: a "daily" series with only ~12 distinct
    # monthly levels stepped across 250 rows.
    monthly = np.repeat(np.arange(13), 20)[:250].astype(float)
    df["us_crude_oil_wti"] = monthly
    with pytest.raises(DataQualityError) as ei:
        validate_feature_matrix(df)
    assert any(i.code == "low_resolution" and i.column == "us_crude_oil_wti"
               for i in ei.value.issues)


def test_infinity_in_a_column_is_a_hard_error():
    # The run #2b failure class: a column builds, isn't all-NaN, but holds +/-inf
    # (double-stationarized indices -> log(log_return) -> -inf) that breaks the
    # scaler. Must be rejected at the matrix, not silently sanitized by models.
    df = _clean_matrix()
    col = df["log_return"].to_numpy().copy()
    col[10] = np.inf
    col[20] = -np.inf
    df["gspc"] = col
    with pytest.raises(DataQualityError) as ei:
        validate_feature_matrix(df, context="AAPL/transformed")
    assert any(i.code == "non_finite" and i.column == "gspc" for i in ei.value.issues)


def test_zero_variance_is_a_warning_not_an_error():
    df = _clean_matrix()
    df["balance_sheet_q_goodwill"] = 0.0  # genuinely flat for some tickers
    issues = validate_feature_matrix(df)  # must NOT raise
    warns = [i for i in issues if i.code == "zero_variance"]
    assert warns and warns[0].column == "balance_sheet_q_goodwill"
    assert all(i.level == "warn" for i in issues)


def test_allow_constant_suppresses_zero_variance_warning():
    df = _clean_matrix()
    df["balance_sheet_q_goodwill"] = 0.0
    issues = validate_feature_matrix(df, allow_constant={"balance_sheet_q_goodwill"})
    assert not any(i.code == "zero_variance" for i in issues)


def test_high_nan_fraction_warns_but_does_not_raise():
    df = _clean_matrix(n=120)
    col = np.full(120, np.nan)
    col[:30] = 1.0  # 75% NaN, not all-NaN
    df["sparse_feature"] = col
    issues = validate_feature_matrix(df)  # must not raise
    assert any(i.code == "high_nan_frac" and i.column == "sparse_feature" for i in issues)


def test_non_datetime_index_is_an_error():
    df = _clean_matrix().reset_index(drop=True)  # RangeIndex
    with pytest.raises(DataQualityError) as ei:
        validate_feature_matrix(df)
    assert any(i.code == "index_not_datetime" for i in ei.value.issues)


def test_too_few_rows_is_an_error():
    with pytest.raises(DataQualityError) as ei:
        validate_feature_matrix(_clean_matrix(n=10), min_rows=50)
    assert any(i.code == "too_few_rows" for i in ei.value.issues)


def test_env_override_downgrades_errors_to_non_raising(monkeypatch):
    df = _clean_matrix()
    df["dead"] = np.nan
    monkeypatch.setenv("ALLOW_DATA_QUALITY_FAIL", "1")
    issues = validate_feature_matrix(df)  # must NOT raise despite the all-NaN error
    assert any(i.code == "all_nan" for i in issues)


def test_non_strict_returns_issues_without_raising():
    df = _clean_matrix()
    df["dead"] = np.nan
    issues = validate_feature_matrix(df, strict=False)
    assert any(i.code == "all_nan" for i in issues)
