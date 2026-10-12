"""
Run #2b stationarized-exog transform table.

The baseline run (#0/#1/#2) feeds the neural nets RAW, non-stationary exogenous
levels (every macro/index feature has ``transform: none``; the z-score scaler does
NOT remove a unit root). This module defines the pre-specified, per-feature
stationarizing transform for the run #2b arm — "give the black boxes
properly-conditioned inputs and see if they still lose."

Design rules (Codex review #2/#4: do NOT blanket-difference — over-differencing an
already-stationary series destroys signal):
  - Positive, exponentially-trending LEVELS  -> ``log_return``  (log-difference).
  - Series that can be ZERO or NEGATIVE (rates, yields, spreads, balances, WTI)
    -> ``diff`` (first difference); log/pct are undefined or unstable there.
  - Borderline mean-reverting series (rates, sentiment, vol, financial-conditions
    indices) are arguably already I(0). We default them to ``diff`` (the finance
    convention treats them as near-unit-root I(1)) but FLAG each one below so the
    thesis can justify or flip it to ``none`` per ADF/KPSS evidence.

This table is the SINGLE SOURCE OF TRUTH. `config.run2b_stationary.yaml` is generated
from it via `stationarize_exog_cfg`; do not hand-edit transform keys in that config.
The univariate AR(1) headline baseline never sees exog, so this arm cannot affect it.
"""

from __future__ import annotations

import copy
from typing import Dict

# Index / ETF price levels — strictly positive, trending -> log-return.
_INDEX_LOG_RETURN = ["^gspc", "^ndx", "^dji", "^gdaxi", "^stoxx50e", "urth", "acwi", "gld"]

# Positive, trending macro LEVELS -> log-return (log-difference = growth rate).
_MACRO_LOG_RETURN = [
    "us_gross_domestic_product",
    "us_consumer_price_index", "us_core_cpi",
    "us_personal_consumption_expenditures", "us_core_pce",
    "us_nonfarm_payrolls", "us_average_hourly_earnings",
    "us_industrial_production_index",
    "us_real_manufacturing_and_trade_sales_inflation_adjusted",
    "us_producer_price_index", "us_broad_retail_sales",
    "us_corporate_profits_after_tax",
    "us_housing_starts", "us_building_permits", "us_existing_home_sales",
    "us_case_shiller_home_price_index",
    "us_m2_money_stock", "us_total_commercial_banks_credit",
    "us_imports", "us_exports",
    "us_weekly_initial_jobless_claims",  # positive flow; spiky but > 0
]

# Rates / yields / spreads / balances -> first difference.
# Persistent, near-unit-root, and can be zero or negative (spreads, trade balance,
# WTI in Apr-2020) where log/pct are undefined.
_MACRO_DIFF = [
    "us_federal_funds_rate",
    "us_treasury_yields_1y", "us_treasury_yields_2y", "us_treasury_yields_5y",
    "us_treasury_yields_10y", "us_treasury_yields_30y",
    "us_corporate_bonds_spread", "us_10y_2y_treasury_spread",
    "us_trade_balance",        # deficit -> negative
    "us_crude_oil_wti",        # went negative Apr-2020 -> log/pct break
]

# BORDERLINE: arguably already I(0) (mean-reverting / bounded). Defaulted to diff
# (finance convention), but FLAGGED — review with ADF/KPSS; flip to "none" if I(0).
# Reviewed 2026-06-25 via scripts/stationarity_methods_check.py on the FIXED loader
# (all 8 keys load; see [[pitfall_fred_freq_inference_dead_features]]): of the 8,
# only us_capacity_utilization_total tested I(0) at the level (ADF p=0.02 reject
# unit root, KPSS p=0.10 fail-to-reject stationarity) -> moved to _MACRO_NONE below
# to avoid over-differencing an already-stationary series. The other 7 stay diff:
# level non-stationary or inconclusive, with diff confirmed I(0) (no over-diff risk).
# unemployment_rate / labor_force_participation are level-inconclusive but
# diff-confirmed I(0) -> diff is the defensible conservative call.
_MACRO_DIFF_BORDERLINE = [
    "us_unemployment_rate",
    "us_labor_force_participation",
    "us_personal_savings_rate",
    "us_consumer_sentiment",
    "us_credit_card_delinquencies",
    "us_chicago_fed_nfci",            # financial conditions index: can be negative
    "us_gold_volatility_cboe_gvz",    # vol index: positive but mean-reverting
]

# Already I(0) at the level per ADF/KPSS -> keep the raw level (no transform).
# Differencing a stationary series over-differences it and destroys signal.
_MACRO_NONE = [
    "us_capacity_utilization_total",  # ADF p=0.02 + KPSS p=0.10 -> stationary level
]

# feature_key -> stationarizing transform name. The single source of truth.
EXOG_TRANSFORM_TABLE: Dict[str, str] = {
    **{k: "log_return" for k in _INDEX_LOG_RETURN},
    **{k: "log_return" for k in _MACRO_LOG_RETURN},
    **{k: "diff" for k in _MACRO_DIFF},
    **{k: "diff" for k in _MACRO_DIFF_BORDERLINE},
    **{k: "none" for k in _MACRO_NONE},
}

# Keys whose classification is a judgment call the thesis must defend (see above).
BORDERLINE_KEYS = frozenset(_MACRO_DIFF_BORDERLINE)


def exog_feature_keys(cfg: dict) -> set:
    """All exogenous (macro + index) feature keys declared in the config."""
    feats = cfg.get("features", {}) or {}
    keys = set((feats.get("macro") or {}).keys())
    keys |= set((feats.get("index") or {}).keys())
    return keys


def stationarize_exog_cfg(cfg: dict) -> dict:
    """Return a deep copy of ``cfg`` with every exogenous feature's ``transform``
    overridden per ``EXOG_TRANSFORM_TABLE``. Internal price/financial features are
    untouched (the close ``log_return`` target and its lagged feature stay as-is).

    Raises ValueError if the config carries an exog feature not in the table — a new
    FRED/index series must get an explicit, reviewed stationarity decision rather
    than silently defaulting to a (possibly wrong) transform.
    """
    out = copy.deepcopy(cfg)
    feats = out.get("features", {}) or {}
    declared = exog_feature_keys(out)
    missing = declared - set(EXOG_TRANSFORM_TABLE)
    if missing:
        raise ValueError(
            "stationarize_exog_cfg: exog feature(s) without a stationarity decision: "
            f"{sorted(missing)}. Add them to EXOG_TRANSFORM_TABLE (with rationale)."
        )
    for group in ("macro", "index"):
        for key, meta in (feats.get(group) or {}).items():
            if isinstance(meta, dict) and key in EXOG_TRANSFORM_TABLE:
                meta["transform"] = EXOG_TRANSFORM_TABLE[key]
    return out
