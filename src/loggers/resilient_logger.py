"""
Resilience wrapper around SupabaseLogger.

If a Supabase call fails (paused free-tier project, network blip, rate limit,
schema mismatch), the experiment must not die mid-run. This wrapper catches
the exception, appends the row to a local JSONL fallback file, and returns a
synthetic stub experiment_id so the calling model code can continue.

Reconcile fallback rows later by reading the JSONL and POSTing manually.
"""

from __future__ import annotations

import json
import os
import time
import uuid as _uuid
from typing import Any, Dict, List, Optional


class ResilientLogger:
    def __init__(self, inner, fallback_dir: str = "results/logger_fallback",
                 quiet_failures: bool = False):
        """
        :param inner: a SupabaseLogger (or any object exposing log_result,
            log_historic_forecast, log_future_forecast).
        :param fallback_dir: directory for the JSONL fallback files.
        :param quiet_failures: if True, suppress repeated per-fold WARN prints
            after the first failure of the run.
        """
        self.inner = inner
        self.fallback_dir = fallback_dir
        self.quiet_failures = quiet_failures
        self._failed_once = False
        os.makedirs(self.fallback_dir, exist_ok=True)

    # ─── Internal helpers ─────────────────────────────────────────────────────

    def _warn(self, table: str, exc: Exception) -> None:
        msg = f"[ResilientLogger] {table} insert failed ({type(exc).__name__}: {exc}); writing to local fallback."
        if not self.quiet_failures or not self._failed_once:
            print(msg)
        self._failed_once = True

    def _fallback_path(self, table: str) -> str:
        return os.path.join(self.fallback_dir, f"{table}.jsonl")

    def _append_jsonl(self, table: str, row: Dict[str, Any]) -> None:
        # Use Supabase's own serializer when available so types match.
        try:
            from loggers.supabase_logger import SupabaseLogger
            serial = SupabaseLogger.to_serializable(row)
        except Exception:
            serial = row
        with open(self._fallback_path(table), "a") as f:
            f.write(json.dumps(serial, default=str) + "\n")

    # ─── Public API (mirrors SupabaseLogger) ──────────────────────────────────

    def log_result(self, **kwargs) -> Optional[str]:
        try:
            return self.inner.log_result(**kwargs)
        except Exception as exc:
            self._warn("experiment_results", exc)
            stub_id = f"fallback-{_uuid.uuid4()}"
            row = dict(kwargs)
            row["_fallback_id"] = stub_id
            row["_fallback_ts"] = time.time()
            self._append_jsonl("experiment_results", row)
            return stub_id

    def log_results_batch(self, rows: List[Dict]) -> int:
        """Batch upsert of per-fold result rows (centralized logging path).

        On any inner failure, write every row to the local experiment_results
        fallback JSONL and return 0, so a paused/erroring Supabase never drops
        fold data mid-run. The inner SupabaseLogger.log_results_batch RAISES on
        error responses (not just exceptions), so this fallback actually fires.
        """
        try:
            return self.inner.log_results_batch(rows)
        except Exception as exc:
            self._warn("experiment_results", exc)
            ts = time.time()
            for r in rows:
                self._append_jsonl("experiment_results", {**r, "_fallback_ts": ts})
            return 0

    def upload_snapshot(self, bucket: str, path: str, data: bytes,
                        content_type: str = "application/octet-stream") -> str:
        """Upload a snapshot; on failure, write it to a local fallback file and
        return that path (soft-fail per eng-review I5 — never crash the grid on
        a transient Storage blip; reconcile the local file later)."""
        try:
            return self.inner.upload_snapshot(bucket, path, data, content_type)
        except Exception as exc:
            self._warn(f"storage:{bucket}/{path}", exc)
            local_dir = os.path.join(self.fallback_dir, "snapshots", os.path.dirname(path))
            os.makedirs(local_dir, exist_ok=True)
            local_path = os.path.join(self.fallback_dir, "snapshots", path)
            with open(local_path, "wb") as f:
                f.write(data)
            return f"local://{local_path}"

    def download_snapshot(self, bucket: str, path: str) -> bytes:
        """Download a snapshot. No fallback: a failure must hard-abort the
        caller (seeds 2-N cannot proceed without the exact group snapshot)."""
        return self.inner.download_snapshot(bucket, path)

    def log_historic_forecast(self, row: Dict) -> Optional[Any]:
        try:
            return self.inner.log_historic_forecast(row)
        except Exception as exc:
            self._warn("experiment_historic_forecasts", exc)
            row = {**row, "_fallback_ts": time.time()}
            self._append_jsonl("experiment_historic_forecasts", row)
            return None

    def log_future_forecast(self, rows: List[Dict]) -> Optional[Any]:
        try:
            return self.inner.log_future_forecast(rows)
        except Exception as exc:
            self._warn("experiment_future_forecasts", exc)
            ts = time.time()
            for r in rows:
                self._append_jsonl("experiment_future_forecasts", {**r, "_fallback_ts": ts})
            return None

    def summary_exists(self, run_id: str, ticker: str) -> bool:
        # Read-only existence check for the crash-resume path. Fail-open: any
        # error (including an inner logger without this method) returns False so
        # the ticker is re-run rather than wrongly skipped.
        try:
            return self.inner.summary_exists(run_id, ticker)
        except Exception:
            return False

    def log_summary(self, row: Dict) -> Optional[Any]:
        try:
            return self.inner.log_summary(row)
        except Exception as exc:
            self._warn("experiment_summaries", exc)
            row = {**row, "_fallback_ts": time.time()}
            self._append_jsonl("experiment_summaries", row)
            return None
