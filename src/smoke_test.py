"""
Focused smoke test for the CT-RNN + Neural ODE integration (T3 verification).

Restricts the run to:
  - ticker: AAPL only
  - experiment.runs: naive + ctrnn + neural_ode, transformed condition only
  - epochs: 5 for CT-RNN and Neural ODE

Bypasses the legacy ``run_experiment`` train/test block entirely; calls
``run_walk_forward_evaluation`` directly with freshly loaded data.

Goal: confirm the wiring works end-to-end before running the full grid.
Expected wallclock: ~5–15 minutes (depends on Yahoo/FRED latency + per-fold
training cost; 23 folds × 3 models).
"""

import os
import time
import yaml

# Reuse the same module-level helpers main.py uses.
from main import (
    get_internal_data,
    get_external_data,
    get_clean_data,
    run_walk_forward_evaluation,
    SUPABASE_URL,
    SUPABASE_KEY,
)
from loggers.supabase_logger import SupabaseLogger
from loggers.resilient_logger import ResilientLogger


class _StubLogger:
    """In-memory logger for smoke tests when Supabase is unreachable."""

    def __init__(self):
        self.results = []
        self.future = []
        self.historic = []

    def log_result(self, **kwargs):
        self.results.append(kwargs)
        return f"stub-{len(self.results)}"

    def log_future_forecast(self, rows):
        self.future.append(rows)

    def log_historic_forecast(self, row):
        self.historic.append(row)


def main():
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    # ─── Smoke-mode overrides (in-memory only; config.yaml untouched) ─────
    cfg["data"]["internal"]["tickers"] = ["AAPL"]
    # Run #2 smoke: a classical baseline, the auto-order SARIMAX, and BOTH CfC
    # wirings — crucially the fc_backbone variant (the proj_size output-shape trap).
    # A finite RMSE for cfc_default_fc proves the loss isn't broadcasting a
    # [batch, units] tensor against a scalar target (eng-review Codex #1).
    cfg["experiment"]["runs"] = [
        {"model": "naive",       "condition": "transformed"},
        {"model": "sarimax_aic", "condition": "transformed"},
        {"model": "cfc", "variant": "default_ncp", "condition": "transformed",
         "params": {"mode": "default", "wiring": "ncp"}},
        {"model": "cfc", "variant": "default_fc", "condition": "transformed",
         "params": {"mode": "default", "wiring": "fc_backbone",
                    "backbone_units": 16, "backbone_layers": 1}},
    ]
    cfg["models"]["CfC"]["epochs"] = 3

    # Single scenario (period, interval).
    scenario = cfg["experiment"]["scenarios"][0]
    period = scenario["period"]
    interval = scenario["interval"]
    ticker = cfg["data"]["internal"]["tickers"][0]

    print(f"\n=== SMOKE: {ticker} ({period} @ {interval}) ===")
    print(f"Runs: {[r['model'] for r in cfg['experiment']['runs']]}")
    print(f"CfC epochs: {cfg['models']['CfC']['epochs']}")

    t0 = time.time()
    print("\n[1/3] Loading raw data ...")
    raw_internal = get_internal_data(ticker=ticker, period=period, interval=interval, cfg=cfg)
    raw_external = get_external_data(period=period, interval=interval, cfg=cfg)
    print(f"      ... done in {time.time() - t0:.1f}s")

    t1 = time.time()
    print("\n[2/3] Cleaning data ...")
    clean_internal, clean_external = get_clean_data(
        raw_internal=raw_internal, raw_external=raw_external, cfg=cfg
    )
    print(f"      ... done in {time.time() - t1:.1f}s")

    t2 = time.time()
    print("\n[3/3] Walk-forward evaluation ...")
    # Try Supabase first; fall back to in-memory stub if the host is unreachable
    # (e.g. project paused). The smoke is about verifying model wiring, not the
    # backend.
    import socket
    use_stub = os.environ.get("SMOKE_STUB_LOGGER", "0") == "1"
    if use_stub:
        print("      SMOKE_STUB_LOGGER=1 set; using in-memory stub logger.")
        logger = _StubLogger()
    else:
        try:
            host = SUPABASE_URL.replace("https://", "").replace("http://", "").split("/")[0]
            socket.gethostbyname(host)
            # Wrap with ResilientLogger so a pause / network blip mid-run doesn't
            # kill the experiment — failed rows land in results/logger_fallback/.
            logger = ResilientLogger(SupabaseLogger(SUPABASE_URL, SUPABASE_KEY),
                                     quiet_failures=True)
            print("      Supabase reachable; using ResilientLogger(SupabaseLogger).")
        except Exception as exc:
            print(f"      [WARN] Supabase unreachable ({exc}); using in-memory stub logger.")
            logger = _StubLogger()
    metrics, primary_sig_table, secondary_sig_table = run_walk_forward_evaluation(
        clean_internal=clean_internal,
        clean_external=clean_external,
        logger=logger,
        ticker=ticker,
        cfg=cfg,
    )
    print(f"      ... done in {time.time() - t2:.1f}s")

    print(f"\n=== SMOKE COMPLETE in {time.time() - t0:.1f}s ===")
    print(f"\nModel keys in metrics: {sorted(metrics.keys())}")
    print(f"Primary DM table rows: {len(primary_sig_table)}")
    if not primary_sig_table.empty:
        print("Models in primary table:",
              sorted(set(primary_sig_table['model_a']) | set(primary_sig_table['model_b'])))

    # Final sanity checks
    expected_keys = {"naive_transformed", "sarimax_aic_transformed",
                     "cfc_default_ncp_transformed", "cfc_default_fc_transformed"}
    actual_keys = set(metrics.keys())
    missing = expected_keys - actual_keys
    if missing:
        print(f"\n[FAIL] Missing model keys in metrics: {missing}")
        raise SystemExit(1)

    for k in expected_keys:
        m = metrics[k]
        if m["n_folds"] == 0:
            print(f"\n[FAIL] {k}: 0 folds completed")
            raise SystemExit(1)
        if not (m["rmse_mean"] == m["rmse_mean"]):  # NaN check
            print(f"\n[FAIL] {k}: RMSE is NaN")
            raise SystemExit(1)
        print(f"  {k}: {m['n_folds']} folds, RMSE={m['rmse_mean']:.6f} "
              f"DA={m['directional_accuracy_mean']:.3f}")

    print("\n[PASS] All smoke checks green.")


if __name__ == "__main__":
    main()
