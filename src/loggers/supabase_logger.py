from supabase import create_client, Client
import numpy as np
import pandas as pd
import datetime
import time
from typing import List, Dict, Optional, Any



class SupabaseLogger:

    """
    Logging experiments in Supabase's PostgresSQL database.
    """

    def __init__(self, url: str, key: str):

        """
        Constructor / Initialising the Supabase client.
        """

        self.supabase: Client = create_client(url, key)



    # ─── HELPER METHODS ───────────────────────────────────────────────────────────────────────────────────────────────

    @staticmethod
    def to_serializable(obj):

        """
        Recursively convert objects to JSON-serializable Python types.
        """

        # numpy / pandas arrays and indexes
        if isinstance(obj, (np.ndarray, pd.Series, pd.Index, list, tuple)):
            return [SupabaseLogger.to_serializable(v) for v in list(obj)]

        # pandas Timestamp / datetime
        if isinstance(obj, pd.Timestamp) or isinstance(obj, datetime.datetime):
            return obj.isoformat()

        # numpy / python floats: reject NaN / Inf -> return None (JSON-safe)
        if isinstance(obj, (float, np.floating)):
            try:
                if not np.isfinite(obj):
                    return None

            except Exception:
                return None

            return float(obj)

        # numpy integer types
        if isinstance(obj, (int, np.integer)):
            return int(obj)

        # numpy bool
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)

        # dict
        if isinstance(obj, dict):
            return {k: SupabaseLogger.to_serializable(v) for k, v in obj.items()}

        # other primitives
        return obj



    def extract_inserted_id(self, response: Any) -> Optional[str]:

        """
        Try to extract the inserted row's uuid from various client response shapes.
        Returns the uuid string if present, else None.
        """

        # Common pattern: response.data is a list of inserted rows (dict-like)
        data = getattr(response, "data", None)

        if data:
            try:
                first = data[0]

                if isinstance(first, dict):
                    for k in ("uuid", "id", "experiment_id"):
                        if k in first:
                            return first[k]

            except Exception:
                pass

        # Some clients return list/dict directly
        if isinstance(response, list) and len(response) > 0 and isinstance(response[0], dict):
            first = response[0]

            for k in ("uuid", "id", "experiment_id"):
                if k in first:
                    return first[k]

        if isinstance(response, dict):
            # maybe wrapped: {"data": [{...}]}
            d = response.get("data")

            if isinstance(d, list) and len(d) > 0 and isinstance(d[0], dict):
                for k in ("uuid", "id", "experiment_id"):
                    if k in d[0]:
                        return d[0][k]

        return None



    # ─── LOGGING RESULTS IN SUPABASE ──────────────────────────────────────────────────────────────────────────────────

    def log_result(self, model_type: str, params: dict, train_start: str, train_end: str, test_start: str,
                   test_end: str, fold_id: int, horizon: int, mse: float, mae: float, rmse: float,
                   directional_accuracy: float, runtime_seconds: float, ticker: str, smape: float, mase: float,
                   notes: str = None, diagnostics: dict = None) -> Optional[str]:

        """
        Appending one row to the experiment_results table.

        :param model_type:
        :param params:
        :param train_start:
        :param train_end:
        :param test_start:
        :param test_end:
        :param fold_id:
        :param horizon:
        :param mse:
        :param mae:
        :param rmse:
        :param smape:
        :param mase:
        :param directional_accuracy:
        :param runtime_seconds:
        :param ticker:
        :param notes:
        :param diagnostics:
        """

        row = {
            "model_type": model_type,
            "params": self.to_serializable(params),
            "train_start": train_start,
            "train_end": train_end,
            "test_start": test_start,
            "test_end": test_end,
            "fold_id": fold_id,
            "horizon": horizon,
            "mse": mse,
            "mae": mae,
            "rmse": rmse,
            "smape": smape,
            "mase": mase,
            "directional_accuracy": directional_accuracy,
            "runtime_seconds": runtime_seconds,
            "ticker": ticker,
            "notes": notes,
            "diagnostics": self.to_serializable(diagnostics) if diagnostics else None
        }

        # perform insert
        response = self.supabase.table("experiment_results").insert(row).execute()

        # robust response handling for different client versions
        try:
            err = getattr(response, "error", None)

            if err:
                print("Supabase returned an error object:", err)
                return None

            # status / status_code check
            status = None

            if hasattr(response, "status_code"):
                status = getattr(response, "status_code")

            elif hasattr(response, "status"):
                status = getattr(response, "status")

            if status is not None:
                if isinstance(status, int) and status >= 400:
                    print("Error inserting into Supabase (status):", status, "response:", repr(response))
                    return None

                else:
                    # try to extract id if possible
                    exp_id = self.extract_inserted_id(response)

                    if exp_id:
                        print("Inserted experiment_results with uuid:", exp_id)
                        return exp_id

                    print("Logged results successfully (status check).")
                    return None

            # handle dict-like response
            if isinstance(response, dict):
                if response.get("error"):
                    print("Error inserting into Supabase:", response.get("error"))
                    return None

                exp_id = self.extract_inserted_id(response)

                if exp_id:
                    print("Inserted experiment_results with uuid:", exp_id)
                    return exp_id

                print("Logged results successfully (dict response).")
                return None

            # fallback: look into .data
            data = getattr(response, "data", None)

            if data is not None:
                try:
                    # try to extract uuid
                    exp_id = self.extract_inserted_id(response)

                    if exp_id:
                        print("Inserted experiment_results with uuid:", exp_id)
                        return exp_id

                except Exception:
                    pass

                print("Logged results successfully. Returned data length:",
                      len(data) if hasattr(data, "__len__") else "unknown")

                return None

            # last resort: print repr for debugging
            print("Insert response (unrecognized object). Inspect this if there are problems:")
            print(repr(response))

            # try extraction anyway
            exp_id = self.extract_inserted_id(response)
            if exp_id:
                print("Inserted experiment_results with uuid (fallback):", exp_id)
                return exp_id

            return None

        except Exception as e:
            print("Exception while handling Supabase response:", e)
            print("Raw response repr:", repr(response))

            # try to extract id in the exception path
            try:
                exp_id = self.extract_inserted_id(response)

                if exp_id:
                    return exp_id

            except Exception:
                pass

            return None



    # Columns that uniquely identify one per-fold result row. Mirrors the
    # full unique index created in migration 002. Used as the upsert arbiter
    # so a crash-resume re-run overwrites the same row instead of duplicating.
    _RESULTS_CONFLICT_COLS = "run_id,ticker,model_type,condition,seed,fold_id"

    def log_results_batch(self, rows: List[Dict]) -> int:
        """
        Upsert a batch of experiment_results rows in a single call.

        Used by the centralized per-fold logging path in main.py. All rows are
        serialized, then upserted with the (run_id, ticker, model_type,
        condition, seed, fold_id) conflict target so re-running the same
        RUN_ID overwrites rather than duplicates (idempotent crash-resume).

        UNLIKE the legacy ``log_result`` (which returns None on an error
        response), this RAISES on any error so the ResilientLogger wrapper
        falls back to the local JSONL instead of silently dropping rows.

        :param rows: list of dicts matching experiment_results columns.
        :returns: number of rows written.
        :raises RuntimeError: on any Supabase error response / non-2xx status.
        """
        if not rows:
            return 0

        serial = [self.to_serializable(r) for r in rows]
        response = (
            self.supabase.table("experiment_results")
            .upsert(serial, on_conflict=self._RESULTS_CONFLICT_COLS)
            .execute()
        )

        # Raise on any signalled error so ResilientLogger.log_results_batch
        # catches it and writes the local fallback (see TODOS "Logger audit":
        # the inner logger must signal failure, not swallow it).
        err = getattr(response, "error", None)
        if err:
            raise RuntimeError(f"experiment_results upsert error: {err}")
        status = getattr(response, "status_code", None) or getattr(response, "status", None)
        if isinstance(status, int) and status >= 400:
            raise RuntimeError(f"experiment_results upsert HTTP {status}: {response!r}")

        return len(serial)

    # ─── Storage (input + predictions snapshots) ─────────────────────────────

    def upload_snapshot(self, bucket: str, path: str, data: bytes,
                        content_type: str = "application/octet-stream",
                        verify: bool = True) -> str:
        """Upload bytes to a Supabase Storage bucket (upsert).

        :param verify: when True (run #2 T4 safeguard), read the object back and
            assert it matches the bytes written. Supabase Storage can return HTTP
            200 yet silently DROP or TRUNCATE the object (e.g. a URL-special char
            in the key) — the run #1 failure where per-seed forecast snapshots
            collided and were lost. A verify mismatch raises, so the ResilientLogger
            wrapper falls back to a local mirror instead of the grid believing the
            snapshot persisted.
        :returns: the bucket/path key on success.
        :raises RuntimeError: on any Storage error or verify mismatch, so
            ResilientLogger falls back to a local file instead of silently losing
            the snapshot.
        """
        try:
            self.supabase.storage.from_(bucket).upload(
                path=path,
                file=data,
                file_options={"content-type": content_type, "upsert": "true"},
            )
        except Exception as exc:
            raise RuntimeError(f"storage upload failed for {bucket}/{path}: {exc}") from exc
        if verify:
            # Read-back with a few retries: read-after-write can transiently 404/503
            # before it settles, and a single failed read would otherwise raise and
            # make ResilientLogger fall back to local — which hard-aborts the
            # load-bearing multi-seed input handoff even though the upload SUCCEEDED.
            readback, last_exc = None, None
            for attempt in range(3):
                try:
                    readback = self.supabase.storage.from_(bucket).download(path)
                    break
                except Exception as exc:
                    last_exc = exc
                    readback = None
                    if attempt < 2:
                        time.sleep(0.5 * (attempt + 1))
            if readback is None:
                raise RuntimeError(
                    f"verify-after-write: read-back failed for {bucket}/{path} after 3 "
                    f"attempts (upload returned OK but the object is unreadable): "
                    f"{last_exc}") from last_exc
            # Byte-equality, NOT just length: a length check would pass a same-size
            # stale object or a key-collision overwrite (the run #1 '#s{seed}' bug
            # could land a different seed's same-size parquet at the key).
            if readback != data:
                raise RuntimeError(
                    f"verify-after-write mismatch for {bucket}/{path}: wrote {len(data)} "
                    f"bytes, read back {len(readback) if readback else 0} and content "
                    "differs. The object was silently dropped, truncated, or overwritten "
                    "by Storage (run #1 failure class).")
        return f"{bucket}/{path}"

    def download_snapshot(self, bucket: str, path: str) -> bytes:
        """Download bytes from a Storage bucket. Raises on failure."""
        try:
            return self.supabase.storage.from_(bucket).download(path)
        except Exception as exc:
            raise RuntimeError(f"storage download failed for {bucket}/{path}: {exc}") from exc

    def log_historic_forecast(self, row: Dict) -> Optional[Any]:

        """
        Insert aggregated in-sample forecast row into experiment_historic_forecasts.
        `row` should be a dict matching the experiment_historic_forecasts columns.
        Returns the client's response or inserted rows when available.

        :param row:
        :returns:
        """

        serial = self.to_serializable(row)
        response = self.supabase.table("experiment_historic_forecasts").insert(serial).execute()

        err = getattr(response, "error", None)

        if err:
            print("Error inserting historic forecast metrics:", err)
            return None

        # try to return inserted uuid if present
        inserted_id = self.extract_inserted_id(response)

        if inserted_id:
            print("Inserted historic forecast with id:", inserted_id)
            return inserted_id

        return response



    def log_future_forecast(self, rows: List[Dict]) -> Optional[Any]:

        """
        Batch-insert many forecast-step rows into experiment_future_forecasts.
        `rows` should be a list of dicts matching the experiment_future_forecasts columns.
        Returns the client's response or inserted rows when available.

        :param rows:
        :returns:
        """

        if not rows:
            print("No forecast rows to insert.")
            return None

        # Ensure everything is serializable
        serializable_rows = [self.to_serializable(r) for r in rows]

        response = self.supabase.table("experiment_future_forecasts").insert(serializable_rows).execute()

        # basic response handling
        err = getattr(response, "error", None)

        if err:
            print("Error inserting forecasts:", err)
            return None

        data = getattr(response, "data", None)

        if data is not None:
            try:
                print(f"Inserted {len(data)} forecast rows.")

            except Exception:
                print("Inserted forecast rows (response has data).")

            return data

        # fallback: try status or repr
        status = None

        if hasattr(response, "status_code"):
            status = getattr(response, "status_code")

        elif hasattr(response, "status"):
            status = getattr(response, "status")

        if status is not None and isinstance(status, int) and status >= 400:
            print("Error inserting forecasts (status):", status, "response:", repr(response))
            return None

        print("Inserted forecast rows (response unknown shape):", repr(response))

        return response



    def summary_exists(self, run_id: str, ticker: str) -> bool:

        """
        Return True iff an experiment_summaries row already exists for this
        (run_id, ticker). Used by the crash-resume path so a restarted run skips
        tickers that already completed.

        Fail-open: any query error (paused project, network blip) returns False
        so the ticker is RE-RUN rather than wrongly skipped. Re-running a done
        ticker at worst duplicates rows; wrongly skipping would lose results.

        :param run_id: the run_id stamped on the summary row.
        :param ticker: the ticker symbol.
        :returns: True if a matching summary row exists, else False.
        """

        if not run_id:
            return False

        try:
            response = (
                self.supabase.table("experiment_summaries")
                .select("uuid")
                .eq("run_id", run_id)
                .eq("ticker", ticker)
                .limit(1)
                .execute()
            )
            data = getattr(response, "data", None)
            return bool(data)

        except Exception as exc:
            print(f"[summary_exists] query failed ({exc}); will re-run {ticker}.")
            return False



    def log_summary(self, row: Dict) -> Optional[Any]:

        """
        Insert a per-(ticker, run) walk-forward summary row into
        experiment_summaries. `row` should already contain the structured
        columns plus the `summary_full` and `fallback_contents` jsonb blobs.

        :param row: dict matching experiment_summaries columns.
        :returns: inserted/updated uuid (str) on success, else the raw response.
        :raises RuntimeError: on any Supabase error response / non-2xx status,
            so ResilientLogger falls back to local JSONL instead of dropping
            the summary silently (see TODOS "Logger audit").
        """

        serial = self.to_serializable(row)
        # Upsert on (run_id, ticker) so a crash-resume re-run of the same RUN_ID
        # overwrites the existing summary instead of inserting a duplicate.
        response = (
            self.supabase.table("experiment_summaries")
            .upsert(serial, on_conflict="run_id,ticker")
            .execute()
        )

        err = getattr(response, "error", None)
        if err:
            raise RuntimeError(f"experiment_summaries upsert error: {err}")
        status = getattr(response, "status_code", None) or getattr(response, "status", None)
        if isinstance(status, int) and status >= 400:
            raise RuntimeError(f"experiment_summaries upsert HTTP {status}: {response!r}")

        inserted_id = self.extract_inserted_id(response)
        if inserted_id:
            print(f"Upserted experiment_summaries for ticker={row.get('ticker')} "
                  f"with uuid: {inserted_id}")
            return inserted_id

        return response
