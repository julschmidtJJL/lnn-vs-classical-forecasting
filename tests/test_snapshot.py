"""
Tests for the input-snapshot + replay path (Workstream A4) and the Storage
methods on Supabase/Resilient loggers.
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.main import (
    _predictions_path, _resolve_df_finals, _safe_object_token, _SNAPSHOT_BUCKET,
)
# Import from the SAME module path main.py resolves (`provenance`, with src on
# sys.path) so FingerprintMismatch is the exact class _resolve_df_finals raises
# — importing it as `src.provenance.FingerprintMismatch` would be a different
# class object and pytest.raises would not catch it.
from provenance import (
    FingerprintMismatch, compute_frame_fingerprint, frame_to_parquet_bytes, sha256_bytes,
)
from src.loggers.resilient_logger import ResilientLogger
from src.loggers.supabase_logger import SupabaseLogger


def _toy_frame():
    idx = pd.date_range("2020-01-01", periods=6, freq="D")
    return pd.DataFrame({"log_return": [0.01, -0.02, 0.0, 0.03, -0.01, 0.02],
                         "feat": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}, index=idx)


class _FakeSnapshotLogger:
    """Serves a known input snapshot (parquet + fingerprint sidecar)."""

    def __init__(self, parquet_bytes, sidecar_dict):
        self._pq = parquet_bytes
        self._sidecar = json.dumps(sidecar_dict, default=str).encode("utf-8")
        self.uploads = []

    def download_snapshot(self, bucket, path):
        if path.endswith(".fingerprint.json"):
            return self._sidecar
        if path.endswith(".parquet"):
            return self._pq
        raise RuntimeError(f"unexpected path {path}")

    def upload_snapshot(self, bucket, path, data, content_type="application/octet-stream"):
        self.uploads.append(path)
        return f"{bucket}/{path}"


def _good_snapshot():
    df = _toy_frame()
    pq = frame_to_parquet_bytes(df)
    sidecar = {"parquet_sha256": sha256_bytes(pq), "fingerprint": compute_frame_fingerprint(df)}
    return df, pq, sidecar


def test_resolve_loads_and_verifies_snapshot():
    df, pq, sidecar = _good_snapshot()
    logger = _FakeSnapshotLogger(pq, sidecar)
    run_meta = {"data_source": "snapshot", "run_group_id": "G"}
    df_finals, fps, uris = _resolve_df_finals(
        logger, {}, {}, {"transformed"}, cfg={}, ticker="AAPL", run_meta=run_meta)
    pd.testing.assert_frame_equal(df_finals["transformed"], df, check_freq=False)
    assert fps["transformed"]["sha256"] == sidecar["fingerprint"]["sha256"]
    assert uris["transformed"].startswith(_SNAPSHOT_BUCKET)


class _PathRecordingSnapshotLogger:
    """Like _FakeSnapshotLogger but records which storage paths were requested,
    so a test can assert reads come from the pinned input group."""

    def __init__(self, parquet_bytes, sidecar_dict):
        self._pq = parquet_bytes
        self._sidecar = json.dumps(sidecar_dict, default=str).encode("utf-8")
        self.requested = []

    def download_snapshot(self, bucket, path):
        self.requested.append(path)
        if path.endswith(".fingerprint.json"):
            return self._sidecar
        if path.endswith(".parquet"):
            return self._pq
        raise RuntimeError(f"unexpected path {path}")


def test_resolve_reads_inputs_from_input_group_when_pinned():
    # run #1/#2 decouple: results write under run_group_id, but inputs are READ
    # from the pinned input_group_id (run #0's frozen snapshot).
    df, pq, sidecar = _good_snapshot()
    logger = _PathRecordingSnapshotLogger(pq, sidecar)
    run_meta = {"data_source": "snapshot", "run_group_id": "RUN1", "input_group_id": "GRID0"}
    df_finals, fps, uris = _resolve_df_finals(
        logger, {}, {}, {"transformed"}, cfg={}, ticker="AAPL", run_meta=run_meta)
    pd.testing.assert_frame_equal(df_finals["transformed"], df, check_freq=False)
    # every requested object lives under the INPUT group, never the run group
    assert logger.requested, "expected snapshot downloads"
    assert all("GRID0/AAPL/" in p for p in logger.requested)
    assert not any("RUN1/" in p for p in logger.requested)
    assert "GRID0/AAPL/" in uris["transformed"]


def test_resolve_reads_from_run_group_when_no_input_group():
    # Backward-compat: without input_group_id, reads come from run_group_id.
    df, pq, sidecar = _good_snapshot()
    logger = _PathRecordingSnapshotLogger(pq, sidecar)
    run_meta = {"data_source": "snapshot", "run_group_id": "RUN1"}
    _resolve_df_finals(logger, {}, {}, {"transformed"}, cfg={}, ticker="AAPL", run_meta=run_meta)
    assert logger.requested
    assert all("RUN1/AAPL/" in p for p in logger.requested)


def test_resolve_hard_aborts_on_parquet_byte_mismatch():
    df, pq, sidecar = _good_snapshot()
    sidecar["parquet_sha256"] = "deadbeef"  # corrupt the byte hash
    logger = _FakeSnapshotLogger(pq, sidecar)
    with pytest.raises(FingerprintMismatch):
        _resolve_df_finals(logger, {}, {}, {"transformed"}, cfg={}, ticker="AAPL",
                           run_meta={"data_source": "snapshot", "run_group_id": "G"})


def test_resolve_hard_aborts_on_fingerprint_mismatch():
    df, pq, sidecar = _good_snapshot()
    sidecar["fingerprint"]["sha256"] = "0" * 64  # parquet byte-hash still ok, content fp wrong
    logger = _FakeSnapshotLogger(pq, sidecar)
    with pytest.raises(FingerprintMismatch):
        _resolve_df_finals(logger, {}, {}, {"transformed"}, cfg={}, ticker="AAPL",
                           run_meta={"data_source": "snapshot", "run_group_id": "G"})


class _LocalFallbackUploadLogger:
    """Simulates Storage being down: upload soft-fails to a local:// path."""

    def upload_snapshot(self, bucket, path, data, content_type="application/octet-stream"):
        return f"local:///tmp/logger_fallback/snapshots/{path}"


def test_resolve_multiseed_aborts_on_local_upload_fallback(monkeypatch):
    # Seeds 2..N DOWNLOAD this snapshot (no download fallback). If seed 1's
    # upload soft-failed to local://, the remote object is missing, so later
    # seeds would hard-abort and a RUN_ID resume can never self-heal (seed 1's
    # summary is already written). The live build must fail fast instead.
    import src.main as main
    monkeypatch.setattr(main, "build_df_final", lambda *a, **k: _toy_frame())
    run_meta = {"data_source": "live", "run_group_id": "G", "multi_seed": True}
    with pytest.raises(RuntimeError, match="could not be uploaded"):
        _resolve_df_finals(_LocalFallbackUploadLogger(), {}, {}, {"transformed"},
                           cfg={}, ticker="AAPL", run_meta=run_meta)


def test_resolve_single_seed_tolerates_local_upload_fallback(monkeypatch):
    # Single-seed run: no later seed ever downloads the snapshot, so a local://
    # fallback is harmless and must NOT abort the run.
    import src.main as main
    monkeypatch.setattr(main, "build_df_final", lambda *a, **k: _toy_frame())
    run_meta = {"data_source": "live", "run_group_id": "G", "multi_seed": False}
    df_finals, fps, uris = _resolve_df_finals(
        _LocalFallbackUploadLogger(), {}, {}, {"transformed"},
        cfg={}, ticker="AAPL", run_meta=run_meta)
    assert "transformed" in df_finals
    assert uris["transformed"].startswith("local://")


# ─── Storage methods on the loggers ──────────────────────────────────────────

class _FakeStorageBucket:
    def __init__(self, fail=False):
        self.fail = fail
        self.uploaded = None
        self._store = {}

    def upload(self, path, file, file_options=None):
        if self.fail:
            raise RuntimeError("storage 500")
        self.uploaded = (path, file)
        # Round-trip so verify-after-write (T4) reads back the exact bytes.
        self._store[path] = file

    def download(self, path):
        if self.fail:
            raise RuntimeError("storage 404")
        # Echo what was uploaded; default for download-only tests.
        return self._store.get(path, b"PARQUET_BYTES")


class _FakeStorage:
    def __init__(self, fail=False):
        self._b = _FakeStorageBucket(fail)

    def from_(self, bucket):
        return self._b


class _FakeSupabaseWithStorage:
    def __init__(self, fail=False):
        self.storage = _FakeStorage(fail)


def _supa_logger(fail=False):
    lg = object.__new__(SupabaseLogger)
    lg.supabase = _FakeSupabaseWithStorage(fail)
    return lg


def test_supabase_upload_ok_and_raises():
    ok = _supa_logger(fail=False)
    assert ok.upload_snapshot("b", "p/x.parquet", b"data") == "b/p/x.parquet"
    bad = _supa_logger(fail=True)
    with pytest.raises(RuntimeError):
        bad.upload_snapshot("b", "p/x.parquet", b"data")


def test_supabase_download_ok_and_raises():
    ok = _supa_logger(fail=False)
    assert ok.download_snapshot("b", "p/x.parquet") == b"PARQUET_BYTES"
    bad = _supa_logger(fail=True)
    with pytest.raises(RuntimeError):
        bad.download_snapshot("b", "p/x.parquet")


def test_resilient_upload_falls_back_to_local_file(tmp_path):
    rl = ResilientLogger(_supa_logger(fail=True), fallback_dir=str(tmp_path))
    uri = rl.upload_snapshot("bucket", "G/AAPL/input_transformed.parquet", b"PARQUET")
    assert uri.startswith("local://")
    local = tmp_path / "snapshots" / "G" / "AAPL" / "input_transformed.parquet"
    assert local.exists() and local.read_bytes() == b"PARQUET"


def test_resilient_download_passes_through_and_raises(tmp_path):
    rl = ResilientLogger(_supa_logger(fail=True), fallback_dir=str(tmp_path))
    with pytest.raises(RuntimeError):
        rl.download_snapshot("bucket", "G/AAPL/input_transformed.parquet")


# ── Storage-key safety (regression for the run #1 '#s{seed}' collision) ──────

def test_safe_object_token_strips_hash_and_unsafe_chars():
    # '#' is the bug that collapsed all run #1 seeds onto one Storage key.
    assert _safe_object_token("grp#s42") == "grp_s42"
    # Allowlist chars survive untouched.
    assert _safe_object_token("run1-logreturn-2026-06-18.v2") == "run1-logreturn-2026-06-18.v2"
    # Other URL-special chars are neutralised too (?, space, %, etc.).
    assert _safe_object_token("a b?c%d") == "a_b_c_d"


def test_predictions_path_distinct_per_seed():
    # Each seed must resolve to a DISTINCT, '#'-free object key so seeds no
    # longer overwrite one another in Supabase Storage.
    paths = {
        _predictions_path("grp", "AAPL", f"grp#s{seed}") for seed in (42, 43, 44, 45, 46)
    }
    assert len(paths) == 5
    assert all("#" not in p for p in paths)
    assert _predictions_path("grp", "AAPL", "grp#s42") == "grp/AAPL/predictions_grp_s42.parquet"
