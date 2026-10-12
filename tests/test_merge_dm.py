"""
Tests for analysis.merge_dm — the optional cross-run DM merge (run #2).

Focus on the pure core: run_key synthesis (run #1 back-compat), the y_true
equality guard (the real pairing check), and that a merged Evaluator spans the
union of variant + baseline keys.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from analysis.merge_dm import (
    ensure_run_key, assert_y_true_consistent, build_evaluator, merge_dm_per_seed,
)
import pytest as _pytest


def _frame(run_key, model, variant, seed=42, ticker="AAPL", n_folds=2, n_t=20,
           y_true=None, pred_noise=0.01, with_run_key=True, seed_rng=0):
    rng = np.random.default_rng(seed_rng)
    rows = []
    for fold_id in range(n_folds):
        yt = y_true[fold_id] if y_true is not None else rng.standard_normal(n_t) * 0.05
        yp = yt + rng.standard_normal(n_t) * pred_noise
        for t in range(n_t):
            row = {
                "run_group_id": "g", "run_id": "g#s42", "seed": seed, "ticker": ticker,
                "model": model, "variant": variant, "condition": "transformed",
                "fold_id": fold_id, "t": t,
                "y_true": float(yt[t]), "y_pred": float(yp[t]), "residual": float(yp[t] - yt[t]),
            }
            if with_run_key:
                row["run_key"] = run_key
            rows.append(row)
    return pd.DataFrame(rows)


# Shared y_true so paired frames agree (the valid case).
_RNG = np.random.default_rng(1)
_SHARED_YT = [_RNG.standard_normal(20) * 0.05, _RNG.standard_normal(20) * 0.05]


def test_ensure_run_key_synthesizes_for_run1_frames():
    """Run #1 parquets lack run_key/variant — synthesize from model+condition."""
    df = _frame(None, "lstm", None, with_run_key=False)
    df = df.drop(columns=["variant"])
    out = ensure_run_key(df)
    assert (out["run_key"] == "lstm_transformed").all()


def test_y_true_consistent_passes_when_shared():
    a = _frame("cfc_pure_ncp_transformed", "cfc", "pure_ncp", y_true=_SHARED_YT, seed_rng=2)
    b = _frame("lstm_transformed", "lstm", None, y_true=_SHARED_YT, seed_rng=3)
    assert_y_true_consistent(pd.concat([a, b], ignore_index=True))  # no raise


def test_y_true_mismatch_hard_fails():
    a = _frame("cfc_pure_ncp_transformed", "cfc", "pure_ncp", y_true=_SHARED_YT, seed_rng=2)
    other_yt = [_SHARED_YT[0], _SHARED_YT[1] + 0.5]  # fold 1 diverges
    b = _frame("lstm_transformed", "lstm", None, y_true=other_yt, seed_rng=3)
    with pytest.raises(ValueError, match="y_true mismatch across run_keys"):
        assert_y_true_consistent(pd.concat([a, b], ignore_index=True))


def test_merge_spans_union_of_keys():
    a = _frame("cfc_pure_ncp_transformed", "cfc", "pure_ncp", y_true=_SHARED_YT, seed_rng=2)
    b = _frame("lstm_transformed", "lstm", None, y_true=_SHARED_YT, seed_rng=3)
    tables = merge_dm_per_seed([a, b])
    assert (42, "AAPL") in tables
    tbl = tables[(42, "AAPL")]
    models = set(tbl["model_a"]) | set(tbl["model_b"])
    assert {"cfc_pure_ncp_transformed", "lstm_transformed"} <= models


def test_build_evaluator_rejects_t_coordinate_mismatch():
    """Within a common fold, run_keys with different t-sets must hard-fail
    (positional misalignment guard) rather than silently compare offset rows."""
    a = _frame("cfc_pure_ncp_transformed", "cfc", "pure_ncp", y_true=_SHARED_YT, seed_rng=2)
    b = _frame("lstm_transformed", "lstm", None, y_true=_SHARED_YT, seed_rng=3)
    # Drop one timestep from b's fold 0 so its t-set differs on a common fold.
    b = b.drop(b[(b["fold_id"] == 0) & (b["t"] == 5)].index)
    df = ensure_run_key(pd.concat([a, b], ignore_index=True))
    with _pytest.raises(ValueError, match="t-coordinates"):
        build_evaluator(df[df["seed"] == 42], ["cfc_pure_ncp_transformed", "lstm_transformed"])


def test_merge_restricts_family_to_primary():
    a = _frame("cfc_pure_ncp_transformed", "cfc", "pure_ncp", y_true=_SHARED_YT, seed_rng=2)
    b = _frame("lstm_transformed", "lstm", None, y_true=_SHARED_YT, seed_rng=3)
    c = _frame("sarimax_transformed", "sarimax", None, y_true=_SHARED_YT, seed_rng=4)
    fam = ["cfc_pure_ncp_transformed", "sarimax_transformed"]
    tables = merge_dm_per_seed([a, b, c], primary_run_keys=fam)
    tbl = tables[(42, "AAPL")]
    models = set(tbl["model_a"]) | set(tbl["model_b"])
    assert models == set(fam)  # lstm excluded from the corrected family
