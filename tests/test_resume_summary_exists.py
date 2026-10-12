"""
Tests for the crash-resume existence check used by main()'s per-ticker skip:
  - SupabaseLogger.summary_exists  (empty run_id short-circuit; fail-open on error)
  - ResilientLogger.summary_exists (passthrough; fail-open on inner error)
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from loggers.supabase_logger import SupabaseLogger
from loggers.resilient_logger import ResilientLogger


class _Resp:
    def __init__(self, data):
        self.data = data


class _FakeTable:
    """Minimal stub of the supabase query builder chain used by summary_exists."""

    def __init__(self, data=None, raise_exc=None):
        self._data = data
        self._raise = raise_exc

    def select(self, *a, **k):
        return self

    def eq(self, *a, **k):
        return self

    def limit(self, *a, **k):
        return self

    def execute(self):
        if self._raise is not None:
            raise self._raise
        return _Resp(self._data)


class _FakeClient:
    def __init__(self, data=None, raise_exc=None):
        self._table = _FakeTable(data=data, raise_exc=raise_exc)

    def table(self, name):
        assert name == "experiment_summaries"
        return self._table


def _logger_with(data=None, raise_exc=None):
    """Build a SupabaseLogger without touching create_client."""
    logger = SupabaseLogger.__new__(SupabaseLogger)
    logger.supabase = _FakeClient(data=data, raise_exc=raise_exc)
    return logger


class TestSupabaseSummaryExists:

    def test_empty_run_id_short_circuits_false(self):
        # No run_id → never claim a prior summary exists (must not query).
        logger = _logger_with(data=[{"uuid": "x"}])
        assert logger.summary_exists("", "AAPL") is False

    def test_existing_row_returns_true(self):
        logger = _logger_with(data=[{"uuid": "abc"}])
        assert logger.summary_exists("run-1", "AAPL") is True

    def test_no_row_returns_false(self):
        logger = _logger_with(data=[])
        assert logger.summary_exists("run-1", "AAPL") is False

    def test_query_error_is_fail_open_false(self):
        # A query failure must NOT skip the ticker → returns False (re-run).
        logger = _logger_with(raise_exc=RuntimeError("paused project"))
        assert logger.summary_exists("run-1", "AAPL") is False


class _InnerRaises:
    def summary_exists(self, run_id, ticker):
        raise RuntimeError("boom")


class _InnerNoMethod:
    pass


class TestResilientSummaryExists:

    def test_passthrough_true(self):
        rl = ResilientLogger(_logger_with(data=[{"uuid": "abc"}]),
                             fallback_dir="results/logger_fallback")
        assert rl.summary_exists("run-1", "AAPL") is True

    def test_inner_error_is_fail_open_false(self):
        rl = ResilientLogger(_InnerRaises(), fallback_dir="results/logger_fallback")
        assert rl.summary_exists("run-1", "AAPL") is False

    def test_inner_without_method_is_fail_open_false(self):
        rl = ResilientLogger(_InnerNoMethod(), fallback_dir="results/logger_fallback")
        assert rl.summary_exists("run-1", "AAPL") is False
