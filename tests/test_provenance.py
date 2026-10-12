"""
Tests for src/provenance.py (Workstream A — reproducibility helpers).
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.provenance import (
    FingerprintMismatch,
    compute_frame_fingerprint,
    env_manifest,
    frame_to_parquet_bytes,
    parquet_bytes_to_frame,
    sha256_bytes,
    verify_frame_against_fingerprint,
)


def _toy_frame():
    idx = pd.date_range("2020-01-01", periods=5, freq="D")
    return pd.DataFrame(
        {"log_return": [0.01, -0.02, 0.0, 0.03, -0.01], "feat": [1.0, 2.0, 3.0, 4.0, 5.0]},
        index=idx,
    )


def test_fingerprint_is_stable():
    df = _toy_frame()
    assert compute_frame_fingerprint(df)["sha256"] == compute_frame_fingerprint(df.copy())["sha256"]


def test_fingerprint_changes_on_value_change():
    df = _toy_frame()
    fp1 = compute_frame_fingerprint(df)["sha256"]
    df2 = df.copy()
    df2.iloc[0, 0] += 1e-6
    assert compute_frame_fingerprint(df2)["sha256"] != fp1


def test_fingerprint_changes_on_column_reorder():
    df = _toy_frame()
    fp1 = compute_frame_fingerprint(df)["sha256"]
    fp2 = compute_frame_fingerprint(df[["feat", "log_return"]])["sha256"]
    assert fp1 != fp2  # column order is part of identity


def test_fingerprint_metadata_fields():
    fp = compute_frame_fingerprint(_toy_frame())
    assert fp["shape"] == [5, 2]
    assert fp["columns"] == ["log_return", "feat"]
    assert fp["nan_counts"] == {"log_return": 0, "feat": 0}
    assert fp["pandas_version"]


def test_nan_counts_tracked():
    df = _toy_frame()
    df.iloc[1, 0] = np.nan
    assert compute_frame_fingerprint(df)["nan_counts"]["log_return"] == 1


def test_parquet_roundtrip_preserves_values_and_fingerprint():
    df = _toy_frame()
    data = frame_to_parquet_bytes(df)
    back = parquet_bytes_to_frame(data)
    # parquet does not round-trip the DatetimeIndex.freq attribute; real
    # trading-day frames have freq=None anyway, and the content fingerprint
    # (which hashes index VALUES, not freq) is what we rely on.
    pd.testing.assert_frame_equal(df, back, check_freq=False)
    # canonical fingerprint survives the parquet roundtrip
    assert compute_frame_fingerprint(df)["sha256"] == compute_frame_fingerprint(back)["sha256"]


def test_byte_hash_changes_with_content():
    df = _toy_frame()
    b1 = frame_to_parquet_bytes(df)
    df2 = df.copy(); df2.iloc[0, 0] += 0.001
    assert sha256_bytes(b1) != sha256_bytes(frame_to_parquet_bytes(df2))


def test_verify_passes_on_match():
    df = _toy_frame()
    verify_frame_against_fingerprint(df, compute_frame_fingerprint(df))  # no raise


def test_verify_hard_aborts_on_mismatch():
    df = _toy_frame()
    fp = compute_frame_fingerprint(df)
    df2 = df.copy(); df2.iloc[0, 0] += 1.0
    with pytest.raises(FingerprintMismatch):
        verify_frame_against_fingerprint(df2, fp)


def test_verify_aborts_when_no_expected_hash():
    with pytest.raises(FingerprintMismatch):
        verify_frame_against_fingerprint(_toy_frame(), {})


def test_fingerprint_rejects_object_columns():
    # Object/string columns hash POINTER ADDRESSES under tobytes(), not values,
    # so they must be refused loudly rather than producing a non-deterministic
    # hash that silently breaks the snapshot replay proof.
    df = _toy_frame()
    df["label"] = ["a", "b", "c", "d", "e"]  # object dtype
    with pytest.raises(TypeError):
        compute_frame_fingerprint(df)


def test_fingerprint_allows_int_and_bool_columns():
    # int/bool are byte-stable; the guard must not reject legitimate numeric data.
    df = _toy_frame()
    df["count"] = [1, 2, 3, 4, 5]
    df["flag"] = [True, False, True, False, True]
    fp1 = compute_frame_fingerprint(df)["sha256"]
    assert fp1 == compute_frame_fingerprint(df.copy())["sha256"]


def test_env_manifest_shape():
    m = env_manifest(seed=42, run_id="R#s42", run_group_id="R")
    assert m["seed"] == 42 and m["run_id"] == "R#s42" and m["run_group_id"] == "R"
    assert m["python_version"] and "numpy" in m["packages"]
