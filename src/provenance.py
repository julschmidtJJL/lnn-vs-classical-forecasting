"""
Reproducibility / provenance helpers (Workstream A).

Two tiers:
  - Lightweight: a CANONICAL fingerprint of an input feature frame (content
    SHA-256 + shape, column order, dtypes, index span/tz, per-column NaN counts,
    pandas/pyarrow versions). Lets us PROVE two runs saw identical inputs without
    storing the data.
  - Full: parquet (de)serialization + byte SHA-256 so the exact input frame can
    be snapshotted to Supabase Storage and replayed offline bit-for-bit.

The content fingerprint hashes the column/dtype/value/index bytes directly
(NOT pandas.util.hash_pandas_object, which is not a stable cross-version
archival hash — see the Codex review). The parquet BYTE hash is stored
separately because parquet bytes can drift by writer/version; on replay we
verify the byte hash first (exact bytes), then the semantic fingerprint.
"""

from __future__ import annotations

import hashlib
import io
import platform
import sys
from importlib import metadata as _ilmd
from typing import Dict

import numpy as np
import pandas as pd


def sha256_bytes(data: bytes) -> str:
    """SHA-256 hex digest of raw bytes."""
    return hashlib.sha256(data).hexdigest()


# numpy dtype kinds whose ``.tobytes()`` is a deterministic function of the
# VALUES: bool, signed/unsigned int, float, datetime64, timedelta64. Object
# ('O') and Python-string arrays hash POINTER ADDRESSES, not content, so their
# bytes differ across processes for identical data — a silent reproducibility
# hole. We refuse them loudly instead (the thesis feature frames are all numeric
# + a datetime index, so this never fires in practice; it guards against a
# non-numeric column silently slipping in and breaking the replay proof).
_BYTE_HASHABLE_KINDS = frozenset("biufMm")


def _assert_byte_hashable(dtype, where: str) -> None:
    kind = np.dtype(dtype).kind
    if kind not in _BYTE_HASHABLE_KINDS:
        raise TypeError(
            f"compute_frame_fingerprint: {where} has dtype {dtype!r} (kind {kind!r}), "
            "whose raw bytes are not a deterministic function of its values "
            "(object/string arrays hash pointer addresses). Convert it to a numeric "
            "or datetime dtype before fingerprinting so snapshots replay reproducibly."
        )


def compute_frame_fingerprint(df: pd.DataFrame) -> Dict:
    """Canonical, content-addressed fingerprint of a feature frame.

    Deterministic given column order, dtypes, cell values and the index — so
    two runs that built identical inputs produce an identical ``sha256``,
    regardless of parquet writer version or machine.
    """
    h = hashlib.sha256()
    for col in df.columns:                       # column order is part of identity
        arr = np.ascontiguousarray(df[col].to_numpy())
        _assert_byte_hashable(arr.dtype, f"column {col!r}")
        h.update(str(col).encode("utf-8"))
        h.update(str(df[col].dtype).encode("utf-8"))
        h.update(arr.tobytes())
    # Index values + name (e.g. the trading-day DatetimeIndex).
    idx_arr = np.ascontiguousarray(df.index.to_numpy())
    _assert_byte_hashable(idx_arr.dtype, "index")
    h.update(str(df.index.name).encode("utf-8"))
    h.update(idx_arr.tobytes())

    tz = getattr(df.index, "tz", None)
    return {
        "sha256": h.hexdigest(),
        "shape": [int(df.shape[0]), int(df.shape[1])],
        "columns": [str(c) for c in df.columns],
        "dtypes": {str(c): str(df[c].dtype) for c in df.columns},
        "index_name": str(df.index.name),
        "index_tz": str(tz) if tz is not None else None,
        "index_start": str(df.index[0]) if len(df.index) else None,
        "index_end": str(df.index[-1]) if len(df.index) else None,
        "nan_counts": {str(c): int(df[c].isna().sum()) for c in df.columns},
        "pandas_version": pd.__version__,
        "pyarrow_version": _pkg_version("pyarrow"),
    }


def frame_to_parquet_bytes(df: pd.DataFrame) -> bytes:
    """Serialize a frame to parquet bytes (index preserved)."""
    buf = io.BytesIO()
    df.to_parquet(buf, engine="pyarrow", index=True)
    return buf.getvalue()


def parquet_bytes_to_frame(data: bytes) -> pd.DataFrame:
    """Inverse of frame_to_parquet_bytes."""
    return pd.read_parquet(io.BytesIO(data), engine="pyarrow")


class FingerprintMismatch(RuntimeError):
    """Raised when a loaded snapshot does not match its stored fingerprint.

    Wrong input data is never tolerated (eng-review I5): callers must hard-abort.
    """


def verify_frame_against_fingerprint(df: pd.DataFrame, expected: Dict) -> None:
    """Raise FingerprintMismatch unless ``df``'s canonical sha256 matches.

    Only the content hash is compared (it already encodes shape/dtype/order/
    values/index); the rest of the dict is descriptive metadata.
    """
    got = compute_frame_fingerprint(df)["sha256"]
    exp = (expected or {}).get("sha256")
    if exp is None:
        raise FingerprintMismatch("no expected sha256 in stored fingerprint")
    if got != exp:
        raise FingerprintMismatch(
            f"snapshot fingerprint mismatch: expected {exp[:12]}…, got {got[:12]}…"
        )


_MANIFEST_PACKAGES = (
    "torch", "numpy", "pandas", "scikit-learn", "scipy", "statsmodels",
    "pyarrow", "yfinance", "fredapi", "supabase",
)


def _pkg_version(name: str):
    try:
        return _ilmd.version(name)
    except Exception:
        return None


def env_manifest(seed=None, run_id=None, run_group_id=None) -> Dict:
    """Runtime environment manifest stored alongside each summary row."""
    return {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "seed": int(seed) if seed is not None else None,
        "run_id": run_id,
        "run_group_id": run_group_id,
        "packages": {p: _pkg_version(p) for p in _MANIFEST_PACKAGES},
    }
