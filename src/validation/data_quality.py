"""
Feature-matrix data-quality gate.

Catches the class of SILENT data-degradation bug that linters and diff review
miss entirely: a column that builds without raising but carries no usable
information. The motivating failure (surfaced 2026-06-25): `clean_fred_series`'
`inferred_freq or "ME"` fallback nulled 4 macro features (CPI, Core CPI,
unemployment, labor-force participation) and downsampled 8 daily series
(treasury yields, WTI, GVZ) to monthly resolution. Both built cleanly and were
silently dropped / degraded before the neural nets ever saw them.

You cannot keep ~24 fill/drop/reindex sites individually verified forever. The
robust defence is to assert on the OUTPUT: one gate over the final feature
matrix catches any of those sites misbehaving, now or in the future.

Severity split (deliberate):
  - ERROR  all-NaN column            -> definitely broken (no data at all)
  - ERROR  +/-inf value in a column  -> breaks downstream scalers; invalid feature
  - ERROR  declared daily series collapsed to coarse resolution
  - ERROR  index not a sorted/unique DatetimeIndex, or too few rows
  - WARN   zero-variance column       -> suspicious but legitimately possible
           (e.g. AAPL goodwill is genuinely 0); surfaced, not fatal
  - WARN   high NaN fraction          -> worth a look, downstream may fill it

`validate_feature_matrix(..., strict=True)` raises `DataQualityError` on any
ERROR-level issue, unless `ALLOW_DATA_QUALITY_FAIL=1` is set (deliberate
throwaway runs), mirroring `ALLOW_NO_GIT_COMMIT` / `ALLOW_PINNED_STATIONARIZED`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd

# Project default: exogenous FRED series that are published at (business-)daily
# cadence and must therefore retain daily resolution in the feature matrix.
# Keys are the sanitized feature keys (see FeatureBuilder.sanitize). Callers may
# override; this default exists so the gate is useful with zero configuration.
DEFAULT_DAILY_EXOG_KEYS = frozenset({
    "us_treasury_yields_1y", "us_treasury_yields_2y", "us_treasury_yields_5y",
    "us_treasury_yields_10y", "us_treasury_yields_30y",
    "us_10y_2y_treasury_spread", "us_crude_oil_wti", "us_gold_volatility_cboe_gvz",
})


@dataclass(frozen=True)
class QualityIssue:
    level: str            # "error" | "warn"
    code: str             # machine-readable: all_nan | low_resolution | high_nan_frac | zero_variance | index_* | too_few_rows
    column: Optional[str] # offending column, or None for matrix-level issues
    detail: str

    def __str__(self) -> str:
        where = f" [{self.column}]" if self.column else ""
        return f"{self.level.upper():5} {self.code}{where}: {self.detail}"


class DataQualityError(RuntimeError):
    """Raised when a strict validation finds one or more ERROR-level issues."""

    def __init__(self, issues: List[QualityIssue], context: str = ""):
        self.issues = issues
        self.context = context
        errs = [i for i in issues if i.level == "error"]
        head = f"data-quality gate FAILED" + (f" ({context})" if context else "")
        body = "\n".join(f"  - {i}" for i in errs)
        super().__init__(
            f"{head}: {len(errs)} error(s).\n{body}\n"
            "Fix the build, or set ALLOW_DATA_QUALITY_FAIL=1 to override for a throwaway run."
        )


def validate_feature_matrix(
    df: pd.DataFrame,
    *,
    context: str = "",
    allow_constant: Iterable[str] = (),
    daily_keys: Iterable[str] = DEFAULT_DAILY_EXOG_KEYS,
    min_step_ratio: float = 0.1,
    max_nan_frac: float = 0.5,
    min_rows: int = 50,
    strict: bool = True,
) -> List[QualityIssue]:
    """Validate a final feature matrix and return the list of issues found.

    :param df: the feature matrix (rows = dates, columns = features).
    :param context: label for error messages (e.g. "AAPL/transformed").
    :param allow_constant: columns allowed to be zero-variance without a warning
        (genuinely-flat features, e.g. AAPL goodwill = 0).
    :param daily_keys: columns that must retain daily resolution; a collapse to
        coarse (monthly) resolution is an ERROR (the downsampling bug class).
    :param min_step_ratio: a daily column whose value changes on fewer than this
        fraction of consecutive observations is flagged as low-resolution. This
        targets the actual pathology (monthly-stepped values change ~1/21 of the
        time) without false-positiving on legitimately low-cardinality daily
        series like basis-point-quoted treasury yields (which revisit levels but
        still change most days).
    :param max_nan_frac: a column with more than this fraction NaN (but not
        all-NaN) is WARNed.
    :param min_rows: matrices with fewer rows are an ERROR.
    :param strict: raise DataQualityError on any ERROR (unless the env override
        ALLOW_DATA_QUALITY_FAIL=1 is set).
    :returns: every issue found (errors and warnings), in order.
    :raises DataQualityError: if strict and there is >=1 ERROR and no override.
    """
    issues: List[QualityIssue] = []
    allow_constant = set(allow_constant)
    daily_keys = set(daily_keys)

    # ── matrix-level: index + size ───────────────────────────────────────────
    n = len(df)
    if n < min_rows:
        issues.append(QualityIssue("error", "too_few_rows", None,
                                   f"{n} rows < min_rows={min_rows}"))
    if not isinstance(df.index, pd.DatetimeIndex):
        issues.append(QualityIssue("error", "index_not_datetime", None,
                                   f"index type is {type(df.index).__name__}, expected DatetimeIndex"))
    else:
        if not df.index.is_monotonic_increasing:
            issues.append(QualityIssue("error", "index_not_sorted", None,
                                       "index is not monotonic increasing"))
        if df.index.has_duplicates:
            dups = int(df.index.duplicated().sum())
            issues.append(QualityIssue("error", "index_duplicates", None,
                                       f"{dups} duplicate index timestamps"))

    # ── per-column ───────────────────────────────────────────────────────────
    for col in df.columns:
        s = df[col]
        non_null = int(s.notna().sum())

        if non_null == 0:
            issues.append(QualityIssue("error", "all_nan", str(col),
                                       "column is entirely NaN (built but carries no data)"))
            continue  # the checks below are meaningless on an empty column

        # Finite-value invariant: a column can build, pass all-NaN/variance checks,
        # yet hold +/-inf that breaks the downstream sklearn scaler ("Input X
        # contains infinity"). This is exactly the run #2b failure: the indices were
        # double-stationarized -> log(log_return) -> -inf. Model-specific inf->0
        # sanitizers then silently corrupt the feature and make results
        # incomparable, so reject the MATRIX here rather than relying on model
        # failures (Codex). Hard error.
        col_vals = pd.to_numeric(s, errors="coerce").to_numpy(dtype="float64")
        n_inf = int(np.isinf(col_vals).sum())
        if n_inf:
            issues.append(QualityIssue("error", "non_finite", str(col),
                                       f"{n_inf} non-finite (+/-inf) value(s) — breaks scalers; "
                                       "feature is invalid, not just noisy"))
            continue  # the per-column checks below are meaningless on an inf-laden column

        nan_frac = 1.0 - non_null / n if n else 0.0
        if nan_frac > max_nan_frac:
            issues.append(QualityIssue("warn", "high_nan_frac", str(col),
                                       f"{nan_frac:.0%} NaN (> {max_nan_frac:.0%})"))

        n_distinct = int(s.nunique(dropna=True))
        if n_distinct <= 1 and str(col) not in allow_constant:
            issues.append(QualityIssue("warn", "zero_variance", str(col),
                                       "column is constant (zero variance) — confirm this is intended"))

        if str(col) in daily_keys:
            sv = s.dropna().to_numpy()
            if len(sv) > 1:
                steps = int((sv[1:] != sv[:-1]).sum())
                step_ratio = steps / (len(sv) - 1)
                if step_ratio < min_step_ratio:
                    issues.append(QualityIssue("error", "low_resolution", str(col),
                                               f"daily series looks downsampled: value changes on "
                                               f"{step_ratio:.3f} of days (< {min_step_ratio}); "
                                               f"{steps} changes over {len(sv)} obs"))

    has_error = any(i.level == "error" for i in issues)
    if strict and has_error and not os.environ.get("ALLOW_DATA_QUALITY_FAIL"):
        raise DataQualityError(issues, context)
    return issues


def format_issues(issues: List[QualityIssue]) -> str:
    """Render issues as a readable block (errors first, then warnings)."""
    if not issues:
        return "  (no issues)"
    errs = [i for i in issues if i.level == "error"]
    warns = [i for i in issues if i.level == "warn"]
    lines = [f"  - {i}" for i in errs] + [f"  - {i}" for i in warns]
    return "\n".join(lines)
