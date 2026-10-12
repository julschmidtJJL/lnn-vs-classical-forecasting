"""
Tests for the run #2 T4 data-safety safeguards:

- verify-after-write: SupabaseLogger.upload_snapshot reads the object back and
  raises if Storage silently dropped/truncated it (the run #1 loss). The raise is
  what lets ResilientLogger fall back to a local mirror.
- local mirror: every forecast snapshot is also written to results/forecasts/.
- predictions reload: predictions_to_fold_map inverts the _persist_predictions
  row-flattening (light enabler for per-unit resume).
- batch-by-family: RUN_MODELS filter narrows the run-list to named model families.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from loggers.supabase_logger import SupabaseLogger


# ── fake Supabase Storage ───────────────────────────────────────────────────────

class _FakeStorageBucket:
    def __init__(self, mode="ok"):
        self.mode = mode          # ok | drop | truncate | download_raises | flaky
        self._store = {}
        self._dl_calls = 0

    def from_(self, bucket):
        return self

    def upload(self, path, file, file_options):
        # "drop" = Storage returns 200 but persists nothing (run #1 class).
        if self.mode != "drop":
            self._store[path] = file
        return {"path": path}

    def download(self, path):
        self._dl_calls += 1
        if self.mode == "download_raises":
            raise RuntimeError("simulated storage download error")
        if self.mode == "flaky" and self._dl_calls == 1:
            # Transient read-after-write blip on the first attempt, then settles.
            raise RuntimeError("simulated transient 404")
        data = self._store.get(path, b"")
        if self.mode == "truncate":
            return data[: max(0, len(data) - 5)]
        return data


class _FakeClient:
    def __init__(self, mode="ok"):
        self.storage = _FakeStorageBucket(mode)


def _logger(mode="ok"):
    lg = SupabaseLogger.__new__(SupabaseLogger)  # bypass create_client
    lg.supabase = _FakeClient(mode)
    return lg


# ── verify-after-write ──────────────────────────────────────────────────────────

def test_verify_after_write_happy_path_returns_key():
    lg = _logger("ok")
    key = lg.upload_snapshot("bucket", "g/t/predictions_r.parquet", b"hello-bytes")
    assert key == "bucket/g/t/predictions_r.parquet"


def test_verify_after_write_raises_when_object_silently_dropped():
    lg = _logger("drop")
    with pytest.raises(RuntimeError, match="verify-after-write mismatch"):
        lg.upload_snapshot("bucket", "g/t/x.parquet", b"some-bytes")


def test_verify_after_write_raises_on_truncation():
    lg = _logger("truncate")
    with pytest.raises(RuntimeError, match="verify-after-write mismatch"):
        lg.upload_snapshot("bucket", "g/t/x.parquet", b"0123456789")


def test_verify_after_write_raises_when_readback_fails():
    lg = _logger("download_raises")
    with pytest.raises(RuntimeError, match="read-back failed"):
        lg.upload_snapshot("bucket", "g/t/x.parquet", b"abc")


def test_verify_false_skips_readback():
    # With verify off, a dropped object does NOT raise (used where a read-back
    # round-trip is not wanted). Confirms the gate is the verify flag.
    lg = _logger("drop")
    key = lg.upload_snapshot("bucket", "g/t/x.parquet", b"abc", verify=False)
    assert key == "bucket/g/t/x.parquet"


def test_verify_retries_transient_readback_then_succeeds():
    # A transient read-after-write blip must NOT abort: the read-back retries and
    # succeeds, so a multi-seed input handoff is not falsely aborted.
    lg = _logger("flaky")
    key = lg.upload_snapshot("bucket", "g/t/x.parquet", b"some-bytes")
    assert key == "bucket/g/t/x.parquet"
    assert lg.supabase.storage._dl_calls >= 2  # proves it retried


def test_verify_is_byte_equality_not_just_length():
    # Same length, different content (e.g. a key-collision overwrite landing a
    # different seed's same-size parquet) must FAIL — length-only would pass it.
    lg = _logger("ok")
    lg.supabase.storage.download = lambda path: b"ZZZZ"  # same length, wrong content
    with pytest.raises(RuntimeError, match="content differs"):
        lg.upload_snapshot("bucket", "g/t/x.parquet", b"data")  # len 4 == 4, bytes differ


# ── local mirror + reload + filter (main.py helpers) ────────────────────────────

def _main():
    import importlib
    return importlib.import_module("main")


def test_local_mirror_writes_forecast_parquet(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    m = _main()
    path = m._mirror_predictions_local(b"PARQUETBYTES", "grp#s42", "AAPL", "grp#s42")
    assert os.path.exists(path)
    with open(path, "rb") as f:
        assert f.read() == b"PARQUETBYTES"
    # key is sanitized: no raw '#' on disk (the run #1 truncation char).
    assert "#" not in os.path.relpath(path, tmp_path)
    assert os.path.relpath(path, tmp_path).startswith(os.path.join("results", "forecasts"))


def test_predictions_to_fold_map_inverts_flattening():
    m = _main()
    df = pd.DataFrame([
        {"run_key": "cfc_default_ncp_transformed", "fold_id": 0, "t": 1, "y_true": 0.2, "y_pred": 0.1, "residual": 0.1},
        {"run_key": "cfc_default_ncp_transformed", "fold_id": 0, "t": 0, "y_true": 0.0, "y_pred": 0.05, "residual": -0.05},
        {"run_key": "sarimax_transformed", "fold_id": 3, "t": 0, "y_true": 1.0, "y_pred": 0.9, "residual": 0.1},
    ])
    out = m.predictions_to_fold_map(df)
    assert set(out) == {"cfc_default_ncp_transformed", "sarimax_transformed"}
    fold0 = out["cfc_default_ncp_transformed"][0]
    # sorted by t: row t=0 first
    assert list(fold0["y_true"]) == [0.0, 0.2]
    assert list(fold0["y_pred"]) == [0.05, 0.1]
    assert 3 in out["sarimax_transformed"]


def test_predictions_to_fold_map_keeps_residuals_aligned_with_nan():
    # Faithful inverse: a missing residual stays as NaN at its position (no dropna),
    # so len(residuals) == len(y_true) and interior positions don't shift.
    m = _main()
    df = pd.DataFrame([
        {"run_key": "k", "fold_id": 0, "t": 0, "y_true": 0.0, "y_pred": 0.1, "residual": -0.1},
        {"run_key": "k", "fold_id": 0, "t": 1, "y_true": 0.2, "y_pred": 0.1, "residual": None},
    ])
    fold = m.predictions_to_fold_map(df)["k"][0]
    assert len(fold["residuals"]) == len(fold["y_true"]) == 2
    assert fold["residuals"][0] == -0.1
    assert np.isnan(fold["residuals"][1])


def test_resume_disabled_for_filtered_family_batch():
    m = _main()
    one = [{"name": "short"}]
    two = [{"name": "short"}, {"name": "long"}]
    assert m._resume_enabled(one, None) is True            # single scenario, no filter
    assert m._resume_enabled(one, "") is True
    assert m._resume_enabled(one, "cfc") is False          # filtered batch -> no auto-skip
    assert m._resume_enabled(two, None) is False           # multi-scenario -> already off


def test_filter_runs_by_models():
    m = _main()
    runs = [
        {"model": "naive", "condition": "transformed"},
        {"model": "cfc", "variant": "default_ncp", "condition": "transformed"},
        {"model": "ltc", "variant": "ncp", "condition": "transformed"},
        {"model": "lstm", "condition": "transformed"},
    ]
    assert len(m._filter_runs_by_models(runs, "cfc,ltc")) == 2
    assert len(m._filter_runs_by_models(runs, "CFC")) == 1           # case-insensitive
    assert m._filter_runs_by_models(runs, "") == runs               # empty -> unchanged
    assert m._filter_runs_by_models(runs, None) == runs
    assert m._filter_runs_by_models(runs, "nonesuch") == []         # no match -> empty
