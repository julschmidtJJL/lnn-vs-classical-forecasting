"""
Tests for the run #2b stationarized-exog transform table (features/stationarize.py)
and the new `diff` transform in FeatureBuilder.

Guards the run #2b arm: every exogenous feature must get an explicit stationarizing
transform (no silent `none` left on a non-stationary level), series that can go
negative must use `diff` (not log/pct), and a newly-added FRED series must fail
loudly until it gets a reviewed decision.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from features.feature_builder import FeatureBuilder
from features import stationarize as st

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "src", "config.yaml")


def _cfg():
    with open(_CONFIG_PATH) as fh:
        return yaml.safe_load(fh)


# ── the `diff` transform ────────────────────────────────────────────────────────

def test_diff_transform_is_first_difference():
    fb = FeatureBuilder(cfg=_cfg())
    s = pd.Series([1.0, 3.0, 6.0, 10.0])
    out = fb.transforms["diff"](s)
    assert np.isnan(out.iloc[0])
    assert list(out.iloc[1:]) == [2.0, 3.0, 4.0]


def test_diff_handles_negative_values():
    # The reason diff exists: log_return/pct_change break on <= 0 (e.g. WTI Apr-2020).
    fb = FeatureBuilder(cfg=_cfg())
    s = pd.Series([5.0, -2.0, -1.0])
    out = fb.transforms["diff"](s)
    assert list(out.iloc[1:]) == [-7.0, 1.0]
    assert np.isfinite(out.iloc[1:]).all()


# ── table coverage ──────────────────────────────────────────────────────────────

def test_table_covers_every_exog_feature_in_config():
    cfg = _cfg()
    declared = st.exog_feature_keys(cfg)
    missing = declared - set(st.EXOG_TRANSFORM_TABLE)
    assert not missing, f"exog features without a stationarity decision: {sorted(missing)}"


def test_table_only_uses_known_transforms():
    fb = FeatureBuilder(cfg=_cfg())
    for key, tname in st.EXOG_TRANSFORM_TABLE.items():
        assert tname in fb.transforms, f"{key} -> unknown transform {tname!r}"


def test_negative_capable_series_use_diff_not_log():
    # Spreads, balances, fed funds and WTI must use diff (can be <= 0).
    for key in ("us_10y_2y_treasury_spread", "us_corporate_bonds_spread",
                "us_trade_balance", "us_crude_oil_wti", "us_chicago_fed_nfci"):
        assert st.EXOG_TRANSFORM_TABLE[key] == "diff"


# ── stationarize_exog_cfg ───────────────────────────────────────────────────────

def test_stationarize_overrides_exog_to_reviewed_transforms():
    cfg = _cfg()
    out = st.stationarize_exog_cfg(cfg)
    feats = out["features"]
    for group in ("macro", "index"):
        for key, meta in (feats.get(group) or {}).items():
            # Every exog gets its REVIEWED transform from the source-of-truth table.
            # The only "none"s allowed are the explicitly-reviewed already-stationary
            # keys (_MACRO_NONE, e.g. capacity_utilization: ADF/KPSS-confirmed I(0) at
            # the level). Any OTHER "none" means a series silently slipped through
            # un-stationarized.
            if meta["transform"] == "none":
                assert key in st._MACRO_NONE, f"{group}.{key} un-stationarized but not a reviewed _MACRO_NONE key"
            assert meta["transform"] == st.EXOG_TRANSFORM_TABLE[key]


def test_reviewed_borderline_decision_capacity_util_none_rest_diff():
    # Locks the 2026-06-25 ADF/KPSS methods decision (run on the FIXED loader, all
    # 8 borderline keys loading): only capacity_utilization tested I(0) at the level
    # -> "none"; the other 7 stay "diff" (level non-stationary/inconclusive, diff
    # confirmed I(0)). Guards the thesis-defended classification against drift.
    assert st._MACRO_NONE == ["us_capacity_utilization_total"]
    assert st.EXOG_TRANSFORM_TABLE["us_capacity_utilization_total"] == "none"
    assert "us_capacity_utilization_total" not in st.BORDERLINE_KEYS
    assert len(st.BORDERLINE_KEYS) == 7
    assert all(st.EXOG_TRANSFORM_TABLE[k] == "diff" for k in st.BORDERLINE_KEYS)


def test_stationarize_does_not_mutate_input_or_internal_features():
    cfg = _cfg()
    before = cfg["features"]["macro"]["us_consumer_price_index"]["transform"]
    out = st.stationarize_exog_cfg(cfg)
    # input cfg untouched (deep copy)
    assert cfg["features"]["macro"]["us_consumer_price_index"]["transform"] == before
    # internal price features (incl. the log_return target) untouched
    assert out["features"]["internal"]["price"]["log_return"]["transform"] == \
        cfg["features"]["internal"]["price"]["log_return"]["transform"]


def test_stationarize_raises_on_unreviewed_exog_feature():
    cfg = _cfg()
    cfg["features"]["macro"]["us_brand_new_series"] = {"unit": "index", "transform": "none"}
    with pytest.raises(ValueError, match="without a stationarity decision"):
        st.stationarize_exog_cfg(cfg)


def test_borderline_keys_are_flagged_subset_of_table():
    assert st.BORDERLINE_KEYS
    assert st.BORDERLINE_KEYS <= set(st.EXOG_TRANSFORM_TABLE)


# ── run #2b config (approach B: config.yaml + one flag) ─────────────────────────

import importlib.util

_RUN2B_PATH = os.path.join(os.path.dirname(__file__), "..", "src", "config.run2b_stationary.yaml")
_GEN_PATH = os.path.join(os.path.dirname(__file__), "..", "scripts", "gen_run2b_config.py")


def _load_gen():
    spec = importlib.util.spec_from_file_location("gen_run2b_config", _GEN_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_run2b_config_exists_and_is_run2b_with_flag():
    with open(_RUN2B_PATH) as fh:
        cfg = yaml.safe_load(fh)
    # Distinct run_kind (review P1) so assert_run_kind can't confuse the arms.
    assert cfg["experiment"]["run_kind"] == "run2b"
    assert cfg["experiment"]["exog_stationarize"] is True


def test_run2b_config_is_base_config_plus_only_run_kind_and_flag():
    # The generated arm must differ from config.yaml ONLY by run_kind (run2->run2b)
    # and the exog_stationarize flag. Run-list, FRED series, seeds, scenarios —
    # everything else identical, so the arm stays comparable to run #2 and never
    # silently drifts.
    base = _cfg()
    with open(_RUN2B_PATH) as fh:
        gen = yaml.safe_load(fh)
    assert gen["experiment"].pop("exog_stationarize") is True
    assert "exog_stationarize" not in base["experiment"]
    assert gen["experiment"]["run_kind"] == "run2b"
    assert base["experiment"]["run_kind"] == "run2"
    gen["experiment"]["run_kind"] = "run2"  # normalize the only other delta
    assert gen == base


def test_base_config_has_no_flag_so_run2_stays_raw_level():
    # The gate in main.py fires on experiment.exog_stationarize. config.yaml must
    # NOT carry it, or run #2 would silently become a stationarized run.
    assert _cfg()["experiment"].get("exog_stationarize", False) is False


def test_gate_contract_flag_on_yields_stationarized_inputs():
    # Mirrors the main.py load-time gate: flag set -> transforms overridden.
    with open(_RUN2B_PATH) as fh:
        cfg = yaml.safe_load(fh)
    assert cfg["experiment"].get("exog_stationarize") is True
    out = st.stationarize_exog_cfg(cfg)
    for group in ("macro", "index"):
        for key, meta in (out["features"].get(group) or {}).items():
            # Inputs are stationarized to the reviewed table; a "none" is only the
            # already-I(0) exceptions (_MACRO_NONE), never an accidental passthrough.
            if meta["transform"] == "none":
                assert key in st._MACRO_NONE
            assert meta["transform"] == st.EXOG_TRANSFORM_TABLE[key]


def test_generator_is_reproducible_and_refuses_double_flag():
    gen = _load_gen()
    with open(os.path.join(os.path.dirname(__file__), "..", "src", "config.yaml")) as fh:
        base_text = fh.read()
    out_text = gen.generate(base_text)
    assert "exog_stationarize: true" in out_text
    # Idempotent intent: feeding an already-flagged config must fail loudly,
    # never double-inject.
    with pytest.raises(ValueError, match="already declares"):
        gen.generate(out_text)


# ── T5 smoke gate configs ───────────────────────────────────────────────────────

_SMOKE_PATH = os.path.join(os.path.dirname(__file__), "..", "src", "config.smoke.yaml")
_SMOKE_STAT_PATH = os.path.join(os.path.dirname(__file__), "..", "src", "config.smoke_stationary.yaml")

# Every new run #2 codepath the smoke must exercise (variant ids in the run-list).
_SMOKE_NEW_VARIANTS = {"ma1", "default_ncp_mm", "default_fc", "ncp_mm"}


def test_smoke_config_exercises_every_new_codepath():
    with open(_SMOKE_PATH) as fh:
        cfg = yaml.safe_load(fh)
    assert cfg["experiment"]["run_kind"] == "smoke"
    variants = {r.get("variant") for r in cfg["experiment"]["runs"]}
    missing = _SMOKE_NEW_VARIANTS - variants
    assert not missing, f"smoke config missing new-codepath variants: {sorted(missing)}"
    # mixed_memory must actually be set on at least one smoke row (not just named).
    assert any((r.get("params") or {}).get("mixed_memory") for r in cfg["experiment"]["runs"])


def test_smoke_stationary_is_smoke_plus_only_the_flag():
    with open(_SMOKE_PATH) as fh:
        base = yaml.safe_load(fh)
    with open(_SMOKE_STAT_PATH) as fh:
        gen = yaml.safe_load(fh)
    assert gen["experiment"].pop("exog_stationarize") is True
    assert "exog_stationarize" not in base["experiment"]
    # run_kind stays "smoke" for both (no separate result group needed).
    assert gen["experiment"]["run_kind"] == base["experiment"]["run_kind"] == "smoke"
    assert gen == base


def test_inject_stationarize_smoke_variant_keeps_run_kind():
    gen = _load_gen()
    src = "experiment:\n  run_kind: smoke\n  seed: 42\n"
    out = gen.inject_stationarize(src, "smoke", "smoke")
    assert "  run_kind: smoke\n" in out
    assert "  exog_stationarize: true\n" in out
    assert out.count("run_kind:") == 1  # value rewritten in place, not duplicated


# ── P0 guard: stationarize is incompatible with a pinned snapshot ───────────────

def _load_main():
    import importlib
    if "main" in sys.modules:
        return sys.modules["main"]
    return importlib.import_module("main")


def test_p0_guard_refuses_stationarize_with_pinned_input_group(monkeypatch):
    main = _load_main()
    monkeypatch.delenv("ALLOW_PINNED_STATIONARIZED", raising=False)
    cfg = {"experiment": {"exog_stationarize": True}}
    with pytest.raises(RuntimeError, match="INPUT_GROUP_ID"):
        main.assert_stationarize_input_source(cfg, "grid-2026-06-11")


def test_p0_guard_allows_stationarize_when_inputs_built_live(monkeypatch):
    main = _load_main()
    monkeypatch.delenv("ALLOW_PINNED_STATIONARIZED", raising=False)
    cfg = {"experiment": {"exog_stationarize": True}}
    main.assert_stationarize_input_source(cfg, None)  # no pin -> live build -> OK


def test_p0_guard_ignores_pinned_input_for_non_stationarized_run(monkeypatch):
    main = _load_main()
    monkeypatch.delenv("ALLOW_PINNED_STATIONARIZED", raising=False)
    cfg = {"experiment": {}}  # run #2 raw-level: pinning to grid-2026-06-11 is fine
    main.assert_stationarize_input_source(cfg, "grid-2026-06-11")


def test_p0_guard_override_env_permits_pinned_stationarized_group(monkeypatch):
    main = _load_main()
    monkeypatch.setenv("ALLOW_PINNED_STATIONARIZED", "1")
    cfg = {"experiment": {"exog_stationarize": True}}
    main.assert_stationarize_input_source(cfg, "some-stationarized-group")


def test_generator_duplicate_check_ignores_comment_mentions():
    # P2 hardening: a comment/doc mentioning the flag in config.yaml must NOT trip
    # the duplicate guard — only a real top-of-line key counts.
    gen = _load_gen()
    base = "experiment:\n  # exog_stationarize: handled by the run2b config\n  run_kind: run2\n"
    out = gen.generate(base)
    assert "  exog_stationarize: true\n" in out
