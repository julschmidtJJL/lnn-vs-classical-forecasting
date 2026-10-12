#!/usr/bin/env python3
"""
Pre-launch data-quality sweep over the feature matrix.

This is the "sweep the whole pipeline" tool: rather than re-audit ~24
fill/drop/reindex sites by hand, it builds the final feature matrix the models
actually train on and asserts it is sane (no all-NaN columns, no daily series
collapsed to monthly, sound index, etc. — see validation/data_quality.py). It
would have caught the 2026-06-25 FRED bug instantly.

Two modes:
  live      (default) build each ticker/condition from FRED+Yahoo and validate.
  snapshot  --snapshot GROUP : pull the stored input snapshots for a run group
            and validate them (e.g. confirm the OLD grid-2026-06-11 snapshot
            still flags the defect, or that a freshly-pinned one is clean).

Exit code is non-zero if any ERROR-level issue is found in any ticker/condition,
so it can gate a launch (the T5 smoke gate calls it).

Usage (needs .env; live mode needs FRED+Yahoo network, snapshot mode needs Supabase):
  python scripts/data_quality_check.py
  python scripts/data_quality_check.py --tickers AAPL SPY JPM
  python scripts/data_quality_check.py --snapshot grid-2026-06-11
"""

from __future__ import annotations

import argparse
import io
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
sys.path.insert(0, str(SRC))
os.chdir(SRC)

import yaml  # noqa: E402
import pandas as pd  # noqa: E402
import main as pipeline  # noqa: E402  (loads .env on import; import-safe)
from validation.data_quality import (  # noqa: E402
    validate_feature_matrix,
    format_issues,
)

CONDITIONS = ("raw", "transformed")


def _build_live(cfg, ticker, period, interval):
    raw_internal = pipeline.get_internal_data(ticker=ticker, period=period, interval=interval, cfg=cfg)
    raw_external = pipeline.get_external_data(period=period, interval=interval, cfg=cfg)
    ci, ce = pipeline.get_clean_data(raw_internal=raw_internal, raw_external=raw_external, cfg=cfg)
    return {cond: pipeline.build_df_final(ci, ce, condition=cond, cfg=cfg, ticker=ticker)
            for cond in CONDITIONS}


def _load_snapshot(logger, group, ticker):
    out = {}
    for cond in CONDITIONS:
        path = pipeline._input_parquet_path(group, ticker, cond)
        data = logger.download_snapshot(pipeline._SNAPSHOT_BUCKET, path)
        out[cond] = pd.read_parquet(io.BytesIO(data))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.yaml", help="config file (relative to src/)")
    ap.add_argument("--tickers", nargs="+", default=None,
                    help="tickers to check (default: all in config.data.internal.tickers)")
    ap.add_argument("--snapshot", metavar="GROUP", default=None,
                    help="validate stored input snapshots for this run group instead of building live")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    # Mirror main.main(): when the config declares the run #2b stationarized-exog
    # arm, apply the transform table at load so we validate the SAME matrix the
    # real run builds (otherwise this would silently check the un-stationarized
    # frame and miss arm-specific degradation).
    if cfg.get("experiment", {}).get("exog_stationarize", False):
        from features.stationarize import stationarize_exog_cfg
        cfg = stationarize_exog_cfg(cfg)
        print("[dq] exog_stationarize=ON: applied EXOG_TRANSFORM_TABLE (run #2b arm).")
    tickers = args.tickers or cfg["data"]["internal"]["tickers"]
    scenario = cfg["experiment"]["scenarios"][0]
    period, interval = scenario["period"], scenario["interval"]

    logger = None
    if args.snapshot:
        from loggers.supabase_logger import SupabaseLogger
        url, key = os.environ.get("SUPABASE_URL"), os.environ.get("SUPABASE_KEY")
        if not (url and key):
            print("ERROR: snapshot mode needs SUPABASE_URL / SUPABASE_KEY in .env")
            return 2
        logger = SupabaseLogger(url, key)
        print(f"[dq] snapshot mode: group={args.snapshot} tickers={tickers}")
    else:
        print(f"[dq] live mode: tickers={tickers} scenario={scenario['name']} ({period}@{interval})")

    total_errors = 0
    total_warns = 0
    for ticker in tickers:
        try:
            frames = (_load_snapshot(logger, args.snapshot, ticker) if args.snapshot
                      else _build_live(cfg, ticker, period, interval))
        except Exception as e:
            print(f"\n=== {ticker}: BUILD/LOAD FAILED: {e} ===")
            total_errors += 1
            continue

        for cond, df in frames.items():
            issues = validate_feature_matrix(df, context=f"{ticker}/{cond}", strict=False)
            errs = [i for i in issues if i.level == "error"]
            warns = [i for i in issues if i.level == "warn"]
            total_errors += len(errs)
            total_warns += len(warns)
            status = "FAIL" if errs else ("warn" if warns else "OK")
            print(f"\n=== {ticker}/{cond}: {df.shape[0]} rows x {df.shape[1]} cols — {status} "
                  f"({len(errs)} errors, {len(warns)} warnings) ===")
            if issues:
                print(format_issues(issues))

    print(f"\n[dq] TOTAL: {total_errors} errors, {total_warns} warnings across "
          f"{len(tickers)} ticker(s).")
    if total_errors:
        print("[dq] DATA-QUALITY GATE RED — do not launch until errors are resolved.")
        return 1
    print("[dq] DATA-QUALITY GATE GREEN.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
