"""
Regression test: SEC fundamentals must be indexed by FILING date, not fiscal
period-end, to avoid lookahead leakage.

Codex flagged (and the AAPL data confirmed) that a 10-Q/10-K is filed ~34 days
after the fiscal period-end. extract_series() previously indexed by "end" and
the feature pipeline forward-filled from there, so models saw a quarter's
fundamentals ~34 days before they were public. Fix: index by "filed", and on
restatements keep the original filing (the value the market actually had).

These tests are network-free (synthetic facts dict) so they run in CI.
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data_import.internal_sec import InternalSECWrapper


def _wrapper():
    return InternalSECWrapper(user_agent="test/1.0", cfg={})


def test_extract_series_indexed_by_filed_not_end():
    facts = {
        "NetIncomeLoss": {
            "units": {
                "USD": [
                    {"end": "2021-09-25", "filed": "2021-10-29", "val": 100, "form": "10-K"},
                    {"end": "2021-12-25", "filed": "2022-01-28", "val": 120, "form": "10-Q"},
                ]
            }
        }
    }
    s = _wrapper().extract_series(facts, "NetIncomeLoss")

    # Index is the filing dates, NOT the period-ends.
    assert list(s.index) == [pd.Timestamp("2021-10-29"), pd.Timestamp("2022-01-28")]
    assert pd.Timestamp("2021-09-25") not in s.index  # period-end must not appear
    assert s.loc[pd.Timestamp("2021-10-29")] == 100


def test_extract_series_keeps_original_filing_on_restatement():
    facts = {
        "NetIncomeLoss": {
            "units": {
                "USD": [
                    # original filing of the Sep-2021 quarter
                    {"end": "2021-09-25", "filed": "2021-10-29", "val": 100, "form": "10-K"},
                    # SAME period restated a year later with a different value
                    {"end": "2021-09-25", "filed": "2022-10-28", "val": 999, "form": "10-K"},
                ]
            }
        }
    }
    s = _wrapper().extract_series(facts, "NetIncomeLoss")

    # Keep the value known at the original filing date; drop the later restatement.
    assert list(s.index) == [pd.Timestamp("2021-10-29")]
    assert s.iloc[0] == 100
    assert 999 not in s.values


def test_extract_series_falls_back_to_end_when_no_filed():
    facts = {
        "NetIncomeLoss": {
            "units": {
                "USD": [
                    {"end": "2021-09-25", "val": 100},
                    {"end": "2021-12-25", "val": 120},
                ]
            }
        }
    }
    s = _wrapper().extract_series(facts, "NetIncomeLoss")
    # No "filed" available -> prior behaviour (index by end).
    assert list(s.index) == [pd.Timestamp("2021-09-25"), pd.Timestamp("2021-12-25")]
