#!/usr/bin/env bash
#
# T5 pre-launch smoke gate — run #2.
#
# Runs 1 ticker (AAPL) / 1 seed over EVERY new run #2 codepath before a real VM
# launch. Each step must exit 0 (and not silently fall back to local snapshots).
# This is a GATE: do not launch the full grid until this is green on the box.
#
# What it covers:
#   0. data-quality      — pre-flight sweep over ALL tickers (AAPL/SPY/JPM) for
#      pre-flight           BOTH arms (raw config.yaml + stationarized
#                           config.run2b_stationary.yaml): validates each built
#                           feature matrix (no all-NaN columns, no daily series
#                           downsampled to monthly, sound index — the FRED
#                           freq-inference bug class). Build+validate only, no
#                           training, so it fails FAST before the expensive
#                           smokes and catches SPY/JPM-specific issues the
#                           AAPL-only training smokes never touch.
#   1. raw smoke        — new model variants (sarimax ma1, cfc mixed_memory,
#                         cfc fc_backbone, ltc mixed_memory) + verify-after-write
#                         + local forecast mirror (both fire on the snapshot writes).
#   2. stationary smoke — the exog_stationarize END-TO-END path: load-time transform
#                         override + diff/log-return exog transforms + feature build
#                         + NN fit on differenced exog.
#   3. filtered smoke   — RUN_MODELS family filter + the per-(ticker,seed) resume
#                         being DISABLED for filtered batches.
#
# Prereqs (on the box): .env with SUPABASE_URL/KEY + FRED_API_KEY, network access.
# Runs from src/ (configs + output_dir are resolved relative to cwd).
#
# Usage:   bash scripts/smoke_gate.sh
# Each step uses a DISTINCT, throwaway RUN_ID so nothing collides with a real run.

set -euo pipefail

cd "$(dirname "$0")/.."/src

_STAMP="smoke-gate-$(date -u +%Y%m%dT%H%M%SZ)"
_pass() { echo "  [PASS] $1"; }
_run() { echo; echo "=== $1 ==="; shift; echo "+ $*"; "$@"; }

echo "T5 smoke gate — $_STAMP (steps 1-3 = 1 ticker / 1 seed; each step must exit 0)"

# 0) Data-quality pre-flight: validate the built feature matrix for ALL tickers,
#    both arms, BEFORE the expensive training smokes. Build+validate only (no
#    model fits), so it fails fast on a silent data defect (e.g. the FRED
#    freq-inference bug) and covers SPY/JPM, which the AAPL-only smokes below do
#    not. Both arms must be GREEN (exit 0); a RED here aborts the gate via set -e.
_run "0/4 data-quality pre-flight (all tickers, raw arm)" \
  python ../scripts/data_quality_check.py --config config.yaml
_pass "data-quality (raw arm, all tickers)"
_run "0/4 data-quality pre-flight (all tickers, stationarized arm)" \
  python ../scripts/data_quality_check.py --config config.run2b_stationary.yaml
_pass "data-quality (stationarized arm, all tickers)"

# 1) Raw smoke: new model variants + verify-after-write + local mirror.
_run "1/4 raw smoke (variants + verify-after-write + mirror)" \
  env RUN_KIND=smoke RUN_ID="${_STAMP}-raw" \
  python main.py config.smoke.yaml
_pass "raw smoke"

# 2) Stationary smoke: exog_stationarize end-to-end (diff transforms + NN fit).
#    INPUT_GROUP_ID MUST be unset (stationarize only applies to a live build).
_run "2/4 stationary smoke (exog_stationarize end-to-end)" \
  env RUN_KIND=smoke RUN_ID="${_STAMP}-stationary" \
  python main.py config.smoke_stationary.yaml
_pass "stationary smoke"

# 3) Filtered smoke: RUN_MODELS family filter + resume-disabled path.
_run "3/4 filtered smoke (RUN_MODELS=cfc, resume disabled)" \
  env RUN_KIND=smoke RUN_ID="${_STAMP}-cfc" RUN_MODELS=cfc \
  python main.py config.smoke.yaml
_pass "filtered smoke"

echo
echo "SMOKE GATE GREEN — data-quality pre-flight + all 3 smokes exited 0. Verify in Supabase that"
echo "experiment_summaries + experiment_results rows exist for the ${_STAMP}-* run_ids"
echo "and that results/forecasts/ has local mirror parquet files, then launch."
