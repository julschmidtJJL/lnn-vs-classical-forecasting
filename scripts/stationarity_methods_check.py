#!/usr/bin/env python3
"""
Stationarity methods check for the run #2b borderline-exog decisions.

`features/stationarize.py` defaults the 8 BORDERLINE_KEYS (unemployment, LFP,
capacity utilization, savings rate, consumer sentiment, CC delinquencies, NFCI,
GVZ) to ``diff`` by finance convention, but the table comment FLAGS each one:
"review with ADF/KPSS; flip to 'none' if I(0)." This script produces exactly that
evidence so the thesis can defend keeping ``diff`` or flipping a series to ``none``.

It runs ADF + KPSS + ACF/PACF on the RAW LEVEL and the first-DIFFERENCE of each
target series, reusing the production data path
(``get_internal_data`` -> ``get_external_data`` -> ``get_clean_data`` ->
``build_df_final(condition="raw")``) and the existing ``DataExplorationCompliance``
toolkit. No new statistics code — same ADF/KPSS/ACF/PACF the rest of the project uses.

Why local files, not Supabase: this is a one-shot METHODS artifact that feeds a
human decision and a thesis table, not experiment output keyed by run_id. It is
committed to the repo so the verdict is version-controlled and `git diff`-able if
the FRED data is ever refreshed.

Outputs (repo-relative):
  results/stationarity/<ticker>/stationarity_report.csv
  results/stationarity/<ticker>/stationarity_report.xlsx
  results/stationarity/<ticker>/plots/<key>__<level|diff>_acf.png   (+ _pacf.png)

Usage (needs .env with FRED_API_KEY + network; Supabase NOT required):
  python scripts/stationarity_methods_check.py                 # AAPL, borderline keys
  python scripts/stationarity_methods_check.py --ticker SPY
  python scripts/stationarity_methods_check.py --keys all-exog  # every exog series
  python scripts/stationarity_methods_check.py --no-plots
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"

# Import the production pipeline. main.py is import-safe (only defs + an
# `if __name__ == "__main__"` guard) and loads the repo-root .env on import.
# Run from src/ so data wrappers resolve any config-relative paths the same way
# the real launch does.
sys.path.insert(0, str(SRC))
os.chdir(SRC)

import main as pipeline  # noqa: E402
from features.stationarize import (  # noqa: E402
    BORDERLINE_KEYS,
    EXOG_TRANSFORM_TABLE,
    exog_feature_keys,
)
from data_exploration.compliance import DataExplorationCompliance  # noqa: E402

ALPHA = 0.05
# Below this many usable observations a verdict is meaningless. Critically, an
# all-NaN series coerces to an EMPTY series, and DataExplorationCompliance treats
# empty/short series as "constant" -> "constant (stationary)". Without this guard
# a series that simply FAILED TO LOAD would be (wrongly) recommended `none`.
MIN_OBS = 30


def _verdict(adf: dict, kpss: dict, alpha: float = ALPHA) -> str:
    """Combine ADF + KPSS into a single verdict — mirrors
    DataExplorationCompliance.stationarity_report so the script's call agrees
    with the pipeline's own reporting."""
    if adf.get("constant") or kpss.get("constant"):
        return "constant (stationary)"
    adf_stationary = (not np.isnan(adf["adf_p"])) and adf["adf_p"] < alpha
    kpss_stationary = (not np.isnan(kpss["kpss_p"])) and kpss["kpss_p"] >= alpha
    if adf_stationary and kpss_stationary:
        return "stationary"
    if (not adf_stationary) and (not kpss_stationary):
        return "non-stationary"
    return "inconclusive"


_STATIONARY_VERDICTS = {"stationary", "constant (stationary)"}


def _analyse_series(comp: DataExplorationCompliance, series: pd.Series) -> dict:
    """ADF + KPSS + ACF/PACF on one (already NaN-dropped) series."""
    s = pd.to_numeric(series, errors="coerce").dropna()
    adf = comp.adfuller_test(s)
    kpss = comp.kpss_test(s)
    acfp = comp.acf_pacf_test(s)
    return {
        "n_obs": len(s),
        "adf_p": adf.get("adf_p"),
        "kpss_p": kpss.get("kpss_p"),
        "verdict": _verdict(adf, kpss),
        "acf_lag1": (float(acfp["acf"][1]) if acfp.get("acf") is not None
                     and len(acfp["acf"]) > 1 else np.nan),
        "n_sig_acf_lags": len(acfp.get("significant_acf_lags") or []),
        "n_sig_pacf_lags": len(acfp.get("significant_pacf_lags") or []),
    }


def _recommend(level_verdict: str, diff_verdict: str, current: str,
               level_n_obs: int) -> tuple[str, str]:
    """Map (level, diff) verdicts to a recommended transform + a note.

    Decision: if the RAW LEVEL is already I(0), prefer ``none`` (avoid
    over-differencing, which destroys signal — Codex review #2/#4). Otherwise
    ``diff``, and confirm the difference actually reaches stationarity."""
    if level_n_obs < MIN_OBS:
        # Empty/too-short series: the "constant (stationary)" verdict here is an
        # artifact of an all-NaN load, NOT real stationarity. Make no decision.
        return "(no decision)", (f"INSUFFICIENT DATA (n_obs={level_n_obs}) — series "
                                 "empty or too short; investigate the load, do NOT flip")
    if level_verdict in _STATIONARY_VERDICTS:
        rec = "none"
        note = "level already I(0)" if level_verdict == "stationary" else "constant level"
    else:
        rec = "diff"
        if diff_verdict in _STATIONARY_VERDICTS:
            note = "level non-stationary, diff confirmed I(0)"
        elif diff_verdict == "inconclusive":
            note = "level non-stationary, diff inconclusive (inspect plots)"
        else:
            note = "diff still non-stationary (investigate — maybe log/2nd-order)"
    changed = "CHANGED" if rec != current else "unchanged"
    return rec, f"{note} [{changed} from default '{current}']"


def _save_plots(outdir: Path, key: str, level: pd.Series, diff: pd.Series) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

    plots = outdir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    for variant, s in (("level", level), ("diff", diff)):
        sv = pd.to_numeric(s, errors="coerce").dropna()
        if len(sv) < 10:
            continue
        lags = min(40, len(sv) // 2 - 1)
        for kind, fn in (("acf", plot_acf), ("pacf", plot_pacf)):
            fig, ax = plt.subplots(figsize=(8, 3.5))
            try:
                fn(sv.values, lags=lags, ax=ax)
                ax.set_title(f"{key} — {variant} — {kind.upper()}")
                fig.tight_layout()
                fig.savefig(plots / f"{key}__{variant}_{kind}.png", dpi=110)
            finally:
                plt.close(fig)


def _resolve_columns(df: pd.DataFrame, keys: set[str]) -> dict[str, str]:
    """Map each target feature key to its actual DataFrame column.

    Macro columns are named by their feature key (e.g. ``us_unemployment_rate``);
    a macro lag may append a suffix. Match exact first, then unique prefix."""
    resolved: dict[str, str] = {}
    for k in sorted(keys):
        if k in df.columns:
            resolved[k] = k
            continue
        candidates = [c for c in df.columns if str(c).startswith(k)]
        if len(candidates) == 1:
            resolved[k] = candidates[0]
        # 0 or >1 candidates -> leave unresolved; reported as missing.
    return resolved


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ticker", default="AAPL", help="ticker to build features for (default AAPL)")
    ap.add_argument("--config", default="config.yaml", help="config file (relative to src/)")
    ap.add_argument("--keys", choices=["borderline", "all-exog"], default="borderline",
                    help="which series to test (default: the 8 borderline keys)")
    ap.add_argument("--outdir", default=None,
                    help="output dir (default: results/stationarity/<ticker>)")
    ap.add_argument("--no-plots", action="store_true", help="skip ACF/PACF PNGs")
    args = ap.parse_args()

    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    scenario = cfg["experiment"]["scenarios"][0]
    period, interval = scenario["period"], scenario["interval"]
    ticker = args.ticker

    print(f"[stationarity] ticker={ticker} scenario={scenario['name']} "
          f"({period}@{interval}) keys={args.keys}")
    print("[stationarity] fetching + cleaning + building RAW-level features "
          "(needs FRED/Yahoo network)...")

    raw_internal = pipeline.get_internal_data(ticker=ticker, period=period, interval=interval, cfg=cfg)
    raw_external = pipeline.get_external_data(period=period, interval=interval, cfg=cfg)
    clean_internal, clean_external = pipeline.get_clean_data(
        raw_internal=raw_internal, raw_external=raw_external, cfg=cfg)
    df_raw = pipeline.build_df_final(clean_internal, clean_external, condition="raw",
                                     cfg=cfg, ticker=ticker)

    target_keys = set(BORDERLINE_KEYS) if args.keys == "borderline" else exog_feature_keys(cfg)
    resolved = _resolve_columns(df_raw, target_keys)
    missing = sorted(target_keys - set(resolved))
    if missing:
        print(f"[stationarity] WARNING: {len(missing)} target key(s) not found as "
              f"columns and skipped: {missing}")
    if not resolved:
        print("[stationarity] ERROR: no target columns resolved — nothing to test.")
        return 2

    comp = DataExplorationCompliance(df_raw)
    outdir = Path(args.outdir) if args.outdir else (REPO_ROOT / "results" / "stationarity" / ticker)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = []
    for key, col in sorted(resolved.items()):
        level = df_raw[col]
        diff = pd.to_numeric(level, errors="coerce").diff()
        lvl = _analyse_series(comp, level)
        dff = _analyse_series(comp, diff)
        current = EXOG_TRANSFORM_TABLE.get(key, "?")
        rec, note = _recommend(lvl["verdict"], dff["verdict"], current, lvl["n_obs"])
        rows.append({
            "key": key,
            "column": col,
            "current_transform": current,
            "recommended_transform": rec,
            "note": note,
            "level_verdict": lvl["verdict"],
            "level_adf_p": lvl["adf_p"],
            "level_kpss_p": lvl["kpss_p"],
            "level_acf_lag1": lvl["acf_lag1"],
            "diff_verdict": dff["verdict"],
            "diff_adf_p": dff["adf_p"],
            "diff_kpss_p": dff["kpss_p"],
            "diff_acf_lag1": dff["acf_lag1"],
            "n_obs": lvl["n_obs"],
        })
        if not args.no_plots:
            _save_plots(outdir, key, level, diff)

    report = pd.DataFrame(rows).set_index("key")
    csv_path = outdir / "stationarity_report.csv"
    xlsx_path = outdir / "stationarity_report.xlsx"
    report.to_csv(csv_path)
    try:
        report.to_excel(xlsx_path)
    except Exception as e:  # openpyxl missing etc. — CSV is the source of truth.
        print(f"[stationarity] (xlsx skipped: {e})")
        xlsx_path = None

    # Console summary: the actionable bit.
    print("\n=== STATIONARITY METHODS CHECK — recommended transforms ===")
    summary = report[["current_transform", "recommended_transform", "level_verdict",
                      "diff_verdict", "note"]]
    with pd.option_context("display.max_colwidth", 60, "display.width", 160):
        print(summary.to_string())
    # Series with no usable data: surface loudly and exclude from the change set.
    no_data = report[report["recommended_transform"] == "(no decision)"]
    if len(no_data):
        print(f"\n!! {len(no_data)} series had NO USABLE DATA (n_obs < {MIN_OBS}) — "
              "NOT a stationarity result, a LOAD problem. Investigate before deciding:")
        for k, r in no_data.iterrows():
            print(f"  - {k} (column '{r['column']}', n_obs={r['n_obs']})")
        # Show which sibling columns DO carry data, to spot a name mismatch.
        for k, r in no_data.iterrows():
            sibs = [(c, int(df_raw[c].notna().sum())) for c in df_raw.columns
                    if str(c).startswith(k.split("__")[0])]
            sibs = [(c, n) for c, n in sibs if n > 0]
            if sibs:
                print(f"    populated columns starting '{k}': {sibs}")
            else:
                print(f"    no populated column starts with '{k}' — series absent from the build.")

    decided = report[report["recommended_transform"] != "(no decision)"]
    changes = decided[decided["current_transform"] != decided["recommended_transform"]]
    print(f"\n{len(changes)} of {len(decided)} DECIDED series would CHANGE from their current default.")
    if len(changes):
        for k, r in changes.iterrows():
            print(f"  - {k}: {r['current_transform']} -> {r['recommended_transform']}  ({r['note']})")
        print("\nNEXT: if you accept a change, edit features/stationarize.py "
              "(move the key out of _MACRO_DIFF_BORDERLINE / set its transform), then\n"
              "      run `python scripts/gen_run2b_config.py` to regenerate both configs.")
    else:
        print("No changes — the finance-convention `diff` default holds for every decided key.")

    print(f"\n[stationarity] wrote {csv_path}")
    if xlsx_path:
        print(f"[stationarity] wrote {xlsx_path}")
    if not args.no_plots:
        print(f"[stationarity] ACF/PACF plots in {outdir / 'plots'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
