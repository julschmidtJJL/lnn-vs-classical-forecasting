#!/usr/bin/env python3
"""
Generate the two exog-robustness configs (audit follow-up, 2026-07-06).

Motivation: the input-data audit (src/docs/audit_findings_2026-07-06.md) found the
SEC fundamental flow features mix quarterly/annual durations (A1) and the FRED macro
features are stale after ~2025-11 (A2). Both defects live in the exogenous block and
are symmetric across models, so the headline should be insensitive to them. This
robustness check tests that directly on ONE cheap cell (AAPL, seed 42) with the two
models the headline claim rests on (fixed-order AR(1) `sarimax` and the run #2
numeric winner `cfc_pure_ncp`) plus the naive baseline:

  config.robust_full.yaml    control arm — full feature set (fresh live build)
  config.robust_noexog.yaml  treatment arm — FRED macro series REMOVED (series: [])
                             and SEC fundamentals REMOVED (features.internal.
                             financial_statement deleted); indices + technicals +
                             price features + past target kept.

Both arms build inputs LIVE (INPUT_GROUP_ID must stay unset) on the same day so the
data window matches; the within-run comparison is (model vs model) per arm and the
across-arm contrast is cfc/sarimax MAE + DM with vs without the defective groups.

Same TEXT-surgery approach as gen_run2b_config.py (comments preserved, zero drift
from config.yaml). Re-run after any change to config.yaml.

Launch (from the box, image built from a commit containing these configs):
  RUN_KIND=run2 RUN_ID=robust-full-20260706    python main.py config.robust_full.yaml
  RUN_KIND=run2 RUN_ID=robust-noexog-20260706  python main.py config.robust_noexog.yaml
"""

from __future__ import annotations

import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, "..", "src")
_IN = os.path.join(_SRC, "config.yaml")
_OUT_FULL = os.path.join(_SRC, "config.robust_full.yaml")
_OUT_NOEXOG = os.path.join(_SRC, "config.robust_noexog.yaml")

_RUNS_BLOCK = """  runs:
    # Exog-robustness cell (audit follow-up): the two headline models + naive.
    - { model: "naive",   condition: "transformed" }
    - { model: "sarimax", condition: "transformed" }                                          # fixed-order AR(1)
    - { model: "cfc", variant: "pure_ncp", condition: "transformed", params: { mode: "pure", wiring: "ncp" } }
"""


def _must_sub(pattern: str, repl: str, text: str, label: str, flags=0) -> str:
    """re.sub that raises if the pattern did not match exactly once — a silent
    no-op surgery would ship a config that LOOKS reduced but is not."""
    new, n = re.subn(pattern, repl, text, count=1, flags=flags)
    if n != 1:
        raise ValueError(f"config surgery '{label}' matched {n} times (expected 1); "
                         "config.yaml structure changed — update this generator.")
    return new


def make_common(text: str) -> str:
    """Shared surgeries for BOTH arms: 1 ticker, 1 seed, 3-model run list."""
    # seeds: [42, 43, 44, 45, 46] -> [42]
    text = _must_sub(r"(?m)^  seeds: \[42, 43, 44, 45, 46\]$",
                     "  seeds: [42]", text, "seeds")
    # tickers: drop SPY + JPM lines (keep AAPL).
    text = _must_sub(r"(?m)^      - SPY .*\n      - JPM .*\n",
                     "", text, "tickers")
    # runs: replace the whole run-list block (from '  runs:' up to the DATA
    # SOURCES banner) with the 3-model robustness list.
    text = _must_sub(r"(?ms)^  runs:\n.*?(?=^# ─── DATA SOURCES)",
                     _RUNS_BLOCK + "\n\n\n", text, "runs block")
    return text


def make_noexog(text: str) -> str:
    """Treatment-arm surgeries: empty the FRED series list, delete the SEC
    financial_statement feature block. Indices/technicals/prices untouched."""
    # fred series list -> [] (from 'series:' up to the '# default lookback' comment).
    text = _must_sub(r"(?ms)^      series:\n.*?(?=^      # default lookback)",
                     "      series: []   # EXOG-ROBUSTNESS ARM: FRED macro removed (audit A1/A2)\n\n",
                     text, "fred series")
    # features.internal.financial_statement -> removed entirely (SEC wrapper then
    # returns empty frames and the feature builder skips fundamentals).
    text = _must_sub(r"(?ms)^    financial_statement:\n.*?(?=^    technical_indicators:)",
                     "    # financial_statement: REMOVED — EXOG-ROBUSTNESS ARM (audit A1/A4)\n\n",
                     text, "financial_statement block")
    return text


def main() -> int:
    with open(_IN) as fh:
        base = fh.read()
    full = make_common(base)
    noexog = make_noexog(full)
    for path, text in [(_OUT_FULL, full), (_OUT_NOEXOG, noexog)]:
        with open(path, "w") as fh:
            fh.write(text)
        print(f"wrote {os.path.relpath(path)} ({text.count(chr(10))} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
