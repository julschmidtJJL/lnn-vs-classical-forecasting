"""
Tests for the centralized per-fold logging path (Workstream B).

Covers:
  - _fold_metrics: matches the inline formulas the models used (finite-mask,
    eps=1e-9, diff-MASE), incl. NaN handling and the no-train-series case.
  - _log_folds_centrally: stamps the REAL walk-forward fold index +
    run_id/condition/seed, includes the naive baseline, and batch-upserts
    one call per (model, condition) run.
  - ResilientLogger.log_results_batch: falls back to local JSONL when the
    inner logger raises (the bundled "Logger audit" fix).
  - SupabaseLogger.log_results_batch / log_summary: RAISE on an error
    response so the resilient wrapper's fallback actually fires.
"""

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.main import _fold_metrics, _log_folds_centrally, _FoldSink, _seed_everything
from src.loggers.resilient_logger import ResilientLogger
from src.loggers.supabase_logger import SupabaseLogger
import pandas as pd


# ─── _fold_metrics ────────────────────────────────────────────────────────────

def test_fold_metrics_matches_model_formula():
    y_true = [0.01, -0.02, 0.0]
    y_pred = [0.012, -0.01, 0.0]
    y_train = [0.01, 0.02, -0.01, 0.03]
    m = _fold_metrics(y_true, y_pred, y_train)
    yt, yp = np.array(y_true), np.array(y_pred)
    assert m["mse"] == pytest.approx(float(np.mean((yt - yp) ** 2)))
    assert m["mae"] == pytest.approx(float(np.mean(np.abs(yt - yp))))
    assert m["rmse"] == pytest.approx(float(np.sqrt(m["mse"])))
    # diff-MASE on the training series
    scale = float(np.mean(np.abs(np.diff(np.array(y_train)))))
    assert m["mase"] == pytest.approx(m["mae"] / (scale + 1e-9))


def test_fold_metrics_drops_nan_predictions():
    # NaN in y_pred must be masked out, not poison the metrics (matches the
    # models' finite_mask behavior). Two finite pairs remain.
    m = _fold_metrics([0.01, 0.02, -0.01], [0.01, np.nan, -0.02], [0.01, 0.02])
    assert m["mae"] == pytest.approx(float(np.mean([0.0, 0.01])))
    assert np.isfinite(m["rmse"])


def test_fold_metrics_all_nan_returns_none():
    m = _fold_metrics([0.01, 0.02], [np.nan, np.nan], [0.01, 0.02])
    assert all(m[k] is None for k in ("mse", "mae", "rmse", "smape", "mase",
                                      "directional_accuracy"))


def test_fold_metrics_no_train_series_mase_none():
    m = _fold_metrics([0.01, -0.02], [0.012, -0.01], y_train=None)
    assert m["mase"] is None
    assert m["mae"] is not None


# ─── _log_folds_centrally ───────────────────────────────────────────────────--

class _RecordingLogger:
    def __init__(self):
        self.calls = []  # list of row-lists

    def log_results_batch(self, rows):
        self.calls.append(rows)
        return len(rows)


def _toy_inputs():
    # 2 folds, initial_window=3, horizon=2 over n=7
    folds = [
        (np.array([0, 1, 2]), np.array([3, 4])),
        (np.array([0, 1, 2, 3, 4]), np.array([5, 6])),
    ]
    df = pd.DataFrame({"log_return": np.linspace(0.01, 0.07, 7),
                       "feat": np.arange(7.0)})
    df_finals = {"transformed": df}
    all_results = {
        "naive_transformed": {
            0: {"y_true": np.array([0.04, 0.05]), "y_pred": np.array([0.03, 0.05]),
                "residuals": np.array([0.01, 0.0])},
            1: {"y_true": np.array([0.06, 0.07]), "y_pred": np.array([0.06, 0.06]),
                "residuals": np.array([0.0, 0.01])},
        },
    }
    return all_results, folds, df_finals


def test_central_logging_stamps_real_fold_index_and_provenance():
    all_results, folds, df_finals = _toy_inputs()
    rec = _RecordingLogger()
    _log_folds_centrally(rec, all_results, folds, df_finals, ticker="AAPL",
                         run_meta={"run_id": "RID#s7"}, seed=7)
    assert len(rec.calls) == 1            # one batch per (model, condition)
    rows = rec.calls[0]
    assert {r["fold_id"] for r in rows} == {0, 1}   # REAL WF fold index, not a global counter
    for r in rows:
        assert r["run_id"] == "RID#s7"
        assert r["condition"] == "transformed"
        assert r["seed"] == 7
        assert r["ticker"] == "AAPL"
        assert r["model_type"] == "Naive"           # naive IS logged now
    # train/test boundaries come from the fold index positions
    f0 = next(r for r in rows if r["fold_id"] == 0)
    assert (f0["train_start"], f0["train_end"], f0["test_start"], f0["test_end"]) == ("0", "2", "3", "4")


def test_central_logging_merges_fold_meta():
    # Per-fold meta (params / runtime_seconds / classical AIC-BIC) captured by a
    # model's meta_fn must land in the experiment_results row, merged with the
    # base params/diagnostics — restoring what the per-model log_result wrote
    # before logging was centralized.
    all_results, folds, df_finals = _toy_inputs()
    all_results["sarimax_transformed"] = {
        0: {"y_true": np.array([0.04, 0.05]), "y_pred": np.array([0.03, 0.05]),
            "residuals": np.array([0.01, 0.0]),
            "meta": {"params": {"order": [1, 1, 1], "seasonal_order": [0, 0, 0, 0]},
                     "runtime_seconds": 1.5,
                     "diagnostics": {"aic": -123.4, "bic": -110.0}}},
    }
    rec = _RecordingLogger()
    _log_folds_centrally(rec, all_results, folds, df_finals, ticker="AAPL",
                         run_meta={"run_id": "R"}, seed=7)
    rows = [r for batch in rec.calls for r in batch]
    sx = next(r for r in rows if r["model_type"] == "SARIMAX")
    assert sx["params"]["order"] == [1, 1, 1]            # meta params merged
    assert sx["params"]["model"] == "SARIMAX"            # base params kept
    assert sx["runtime_seconds"] == 1.5
    assert sx["diagnostics"]["aic"] == -123.4            # classical AIC/BIC carried
    assert sx["diagnostics"]["bic"] == -110.0
    assert sx["diagnostics"]["mse"] is not None          # base metric diagnostics kept
    # naive has no meta → runtime_seconds None, params has no 'order'
    nv = next(r for r in rows if r["model_type"] == "Naive")
    assert nv["runtime_seconds"] is None
    assert "order" not in nv["params"]


def test_central_logging_skips_empty_results():
    rec = _RecordingLogger()
    _log_folds_centrally(rec, {"lstm_transformed": {}}, [], {"transformed": None},
                         ticker="AAPL", run_meta={"run_id": "R"}, seed=1)
    assert rec.calls == []


def test_fold_sink_is_noop():
    assert _FoldSink().log_result(model_type="X", mae=1.0) is None


# ─── Determinism (eng-review I6 — tolerance-based) ────────────────────────────

def test_seed_everything_makes_rng_reproducible():
    import random
    import torch

    def draw():
        _seed_everything(123)
        return (random.random(), float(np.random.rand()), float(torch.rand(1).item()))

    a, b = draw(), draw()
    # Same seed → identical draws across Python/NumPy/Torch RNGs, within a tight
    # tolerance (the pipeline reports per-seed results, so seeding must hold).
    assert a == pytest.approx(b, abs=1e-12)


def test_seed_everything_differs_across_seeds():
    import torch
    _seed_everything(1)
    x1 = float(torch.rand(1).item())
    _seed_everything(2)
    x2 = float(torch.rand(1).item())
    assert x1 != x2


# ─── ResilientLogger fallback on batch error (Logger-audit fix) ───────────────

class _RaisingInner:
    def log_results_batch(self, rows):
        raise RuntimeError("supabase down")


def test_resilient_batch_falls_back_to_jsonl(tmp_path):
    rl = ResilientLogger(_RaisingInner(), fallback_dir=str(tmp_path))
    n = rl.log_results_batch([{"ticker": "AAPL", "fold_id": 0},
                              {"ticker": "AAPL", "fold_id": 1}])
    assert n == 0
    fallback = tmp_path / "experiment_results.jsonl"
    assert fallback.exists()
    lines = fallback.read_text().strip().splitlines()
    assert len(lines) == 2
    rec = json.loads(lines[0])
    assert rec["ticker"] == "AAPL" and "_fallback_ts" in rec


# ─── SupabaseLogger raises on error response (so fallback fires) ──────────────

class _FakeResp:
    def __init__(self, error=None, status_code=200, data=None):
        self.error = error
        self.status_code = status_code
        self.data = data or []


class _FakeChain:
    def __init__(self, resp):
        self._resp = resp

    def upsert(self, *a, **k):
        return self

    def execute(self):
        return self._resp


class _FakeClient:
    def __init__(self, resp):
        self._resp = resp

    def table(self, name):
        return _FakeChain(self._resp)


def _logger_with(resp):
    lg = object.__new__(SupabaseLogger)   # bypass create_client
    lg.supabase = _FakeClient(resp)
    return lg


def test_batch_upsert_raises_on_error_response():
    lg = _logger_with(_FakeResp(error={"message": "boom"}))
    with pytest.raises(RuntimeError):
        lg.log_results_batch([{"ticker": "AAPL", "fold_id": 0}])


def test_batch_upsert_raises_on_http_error_status():
    lg = _logger_with(_FakeResp(status_code=403))
    with pytest.raises(RuntimeError):
        lg.log_results_batch([{"ticker": "AAPL", "fold_id": 0}])


def test_batch_upsert_empty_is_noop():
    lg = _logger_with(_FakeResp())
    assert lg.log_results_batch([]) == 0


def test_log_summary_raises_on_error_response():
    lg = _logger_with(_FakeResp(error={"message": "boom"}))
    with pytest.raises(RuntimeError):
        lg.log_summary({"ticker": "AAPL", "run_id": "R"})
