"""Tests for the chapter-grade expressivity aggregation
(src/analysis/run_expressivity_chapter.py).

Pure-function units (windowing, standardization) plus a tiny end-to-end run with
materialize_features + output dir monkeypatched, so there is no network or write
to the real results directory.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.analysis import run_expressivity_chapter as rec


# ─── nonoverlap_slices ───────────────────────────────────────────────────────

def test_nonoverlap_slices_tail_aligned_and_disjoint():
    sl = rec.nonoverlap_slices(2555, 100)
    assert len(sl) == 25                       # 2555 // 100
    assert sl[-1].stop == 2555                 # tail-aligned
    assert sl[0].start == 2555 - 25 * 100      # oldest remainder dropped
    # non-overlapping and contiguous
    for a, b in zip(sl, sl[1:]):
        assert a.stop == b.start
        assert b.stop - b.start == 100


def test_nonoverlap_slices_shorter_than_window_is_empty():
    assert rec.nonoverlap_slices(50, 100) == []


# ─── standardize_window ──────────────────────────────────────────────────────

def test_standardize_window_drops_target_and_zscores():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "log_return": rng.standard_normal(60),       # target -> must be dropped
        "a": np.cumsum(rng.standard_normal(60)),
        "b": np.cumsum(rng.standard_normal(60)),
    })
    mat = rec.standardize_window(df, slice(0, 60))
    assert mat.shape == (60, 2)                       # log_return dropped
    assert np.allclose(mat.mean(axis=0), 0.0, atol=1e-9)
    assert np.allclose(mat.std(axis=0), 1.0, atol=1e-9)


def test_standardize_window_drops_columns_with_nan_in_window():
    df = pd.DataFrame({
        "a": np.linspace(0, 1, 40),
        "nanny": [np.nan] * 20 + list(range(20)),     # NaN inside the window
    })
    mat = rec.standardize_window(df, slice(0, 40))
    assert mat.shape == (40, 1)                        # nanny dropped


# ─── per-window metrics ──────────────────────────────────────────────────────

def test_canonical_metrics_keys_and_finite():
    rng = np.random.default_rng(1)
    canon_in = rng.standard_normal((2, 60))           # (m=2, steps)
    out = rec.canonical_metrics(canon_in, seed=1, k=16)
    assert set(out) == {"node", "ctrnn", "ltc"}
    for m in out.values():
        assert set(m) == {"raw", "l_norm", "finite"}


def test_bench_metrics_covers_four_archs():
    rng = np.random.default_rng(2)
    X = rng.standard_normal((40, 5))                   # (steps, n_features)
    out = rec.bench_metrics(X, seed=1, units=16)
    assert set(out) == {"ltc", "cfc", "ctrnn", "lstm"}
    for m in out.values():
        assert set(m) == {"raw", "l_norm", "finite"}


# ─── end-to-end (monkeypatched, no network / no real writes) ─────────────────

def _toy_frame(n=250, seed=0):
    rng = np.random.default_rng(seed)
    data = {"log_return": rng.standard_normal(n)}
    for c in "abcdef":
        data[c] = np.cumsum(rng.standard_normal(n))   # features with variance
    return pd.DataFrame(data)


def test_run_chapter_end_to_end(monkeypatch, tmp_path):
    monkeypatch.setattr(rec, "materialize_features", lambda cfg, ticker, force=False: _toy_frame())
    monkeypatch.setattr(rec, "_load_config", lambda: {})
    monkeypatch.setattr(rec, "_OUT_DIR", tmp_path)

    out = rec.run_chapter(["AAPL"], seeds=2, widths=[16], steps=50, units=16,
                          fetch=True, do_bench=True)

    canon = out["canonical"]
    assert set(canon["model"]) == {"node", "ctrnn", "ltc"}
    assert (canon["n"] > 0).all()
    assert np.isfinite(canon["lnorm_median"]).all()

    bench = out["bench"]
    assert set(bench["model"]) == {"ltc", "cfc", "ctrnn", "lstm"}

    assert (tmp_path / "chapter_canonical_summary.csv").exists()
    assert (tmp_path / "chapter_bench_summary.csv").exists()
    assert (tmp_path / "chapter_canonical_per_window.csv").exists()
