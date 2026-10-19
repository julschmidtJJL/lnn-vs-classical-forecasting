---
name: lnn-benchmark
description: Reproduce or extend the liquid-neural-network forecasting benchmark in this repository — set it up, check the headline result against the committed data, run the quick end-to-end check, and add a new model or ticker under the same walk-forward and Diebold–Mariano protocol. Use when someone asks to run, reproduce, check or extend this benchmark.
allowed-tools:
  - Bash
  - Read
  - Write
  - Edit
---

# Reproduce or extend the LNN benchmark

This repository is the experiment behind a bachelor's thesis: 11 liquid
variants (LTC, CfC) against 8 baselines on AAPL, JPM and SPY daily log
returns, expanding-window walk-forward, Diebold–Mariano tests with the
Harvey–Leybourne–Newbold correction and Holm adjustment. The headline is a
null result: 0 of 120 corrected comparisons significant. Read the rules, then
work through the steps in order. Stop and ask whenever a step needs a decision
only the user can make.

## Rules

1. **The thesis results are frozen.** Do not edit anything under
   `src/docs/` (results notes, figures, figure data), the thesis PDF, or the
   numbers in `README.md`. New runs produce new output files; they never
   overwrite or "update" the published ones. If a new run disagrees with a
   published number, report the disagreement; do not reconcile it.
2. **Same protocol for anything new.** A new model or ticker goes through the
   existing walk-forward splitter, the existing metrics and the existing
   DM/Holm code. Do not add early stopping, per-model tuning or a different
   test unless the user asks, and then say plainly that the comparison is no
   longer like-for-like. Every neural model in the thesis gets the same fixed
   epoch budget from the config.
3. **Never handle a secret.** The FRED API key is the user's own. Give them
   the command to put it in `.env`; do not ask them to paste it into the
   chat, and never print `.env`.
4. **Stay inside this repository.** Python packages go into `.venv/` in the
   repository. Do not install anything globally or edit any agent
   configuration. Do not commit or push without asking.
5. **Report what the numbers say.** A smoke run (3 folds, 1 seed, short
   training) is an infrastructure check, not evidence. Never present a smoke
   result as a finding, and never write that a new model "beats" a baseline
   unless the corrected DM test says so on the full protocol.

Tell the user up front what a run needs: Python 3.11, about 1.5 GB of disk for
the dependencies, network access to Yahoo Finance, FRED and SEC EDGAR, and a
free FRED API key. It runs on CPU; no GPU is needed. Supabase is optional.

## 1. Install

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt pytest
.venv/bin/python -m pytest -q tests
```

The test suite is offline and takes a few minutes on a laptop CPU. All tests
should pass before you change anything.

## 2. Check the headline against the committed data

No download, no key, a few seconds:

```bash
.venv/bin/python - <<'EOF'
import pandas as pd
dm = pd.read_csv("src/docs/figures/data/dm_cfc_vs_sarimax_audit.csv")
print(f"{int(dm.significant.sum())} of {len(dm)} corrected comparisons significant "
      f"(min p_corrected = {dm.p_corrected.min()})")
mae = pd.read_csv("src/docs/figures/data/mae_run2a.csv")
print(mae.sort_values(["ticker", "mae_mean"]).groupby("ticker").head(5).to_string(index=False))
EOF
```

This reads the tables the thesis figures were built from. It shows that the
best liquid variant (`cfc_pure_ncp`) and the ARMA baselines sit level, and
that none of the 120 comparisons is significant.

## 3. FRED key

Ask the user to get a free key at https://fred.stlouisfed.org/docs/api/api_key.html
and to create `.env` themselves:

```bash
cp .env.example .env
```

then set `FRED_API_KEY=…` in it with an editor, and `SEC_USER_AGENT` to
their own name and email (SEC EDGAR asks every caller to identify itself; the
template's placeholder overrides the config default). Wait until they say it
is done. The Alpha Vantage, FMP and Finnhub keys in `.env.example` are not
used.

Results normally go to a Supabase project. Without one, set the two Supabase
variables to an address where nothing listens, so every write fails at once
and lands in the local fallback under `src/results/logger_fallback/`:

```
SUPABASE_URL=http://127.0.0.1:9
SUPABASE_KEY=offline
```

The `[ResilientLogger] ... Connection refused; writing to local fallback`
lines this prints are expected.

## 4. Quick end-to-end run

```bash
cd src && ../.venv/bin/python main.py config.smoke.yaml
```

AAPL only, 3 walk-forward folds, one seed, shortened training; about 8
minutes on a 2020 Intel MacBook. It downloads the data, builds both
preprocessing conditions, fits every model family and runs the DM/Holm
tests. The summary lands in
`src/results/walk_forward/AAPL_wf_summary.json`; show the user the metrics
and the primary significance table, and remind them it is a smoke result.

The full grid (`config.yaml`: 3 tickers, 10 years, 19 models, 5 seeds, 50
epochs per neural model and fold) takes days on a CPU. Do not start it
without asking; on a laptop, suggest running one ticker and one seed first,
or a rented Linux VM (`CLOUD_RUN.md`, `Dockerfile`).

## 5. Extend: a new ticker

1. Add it under `data.internal.tickers` in a **copy** of the config
   (`config.smoke.yaml` first, then `config.yaml`). Name the copy after the
   change, e.g. `config.smoke.qqq.yaml`.
2. Check for target leakage: if the ticker is an index fund, its own index is
   among the macro features. Add it to `data.feature_exclusions`, the way
   `SPY` excludes `^GSPC` (a `QQQ` run would exclude `^NDX`). Ask the user if
   unsure.
3. Run the smoke copy, then report.

## 6. Extend: a new model

1. Read `src/models/naive_model.py` (the smallest model) and the model you are
   closest to, e.g. `src/models/cfc_model.py` for a neural one. A model is a
   class constructed with `(cfg, features, logger, ticker)`; neural models
   expose `fit(X_train, y_train, params)` and `predict(...)` returning
   predictions and residuals for one fold.
2. Add `src/models/<name>_model.py`, import it in `src/main.py` next to the
   other models, and add an `elif model == "<name>":` branch in the run loop
   modelled on the closest existing branch. Predictions must be one step
   ahead and must not see the target of the day they predict (see the comment
   above the dispatch in `main.py`).
3. Add its hyperparameters under `models:` in the config copy, with the same
   epoch budget as the other neural models, and one line under
   `experiment.runs`.
4. Add a test in `tests/` modelled on `tests/test_naive_baseline.py` or
   `tests/test_ctrnn_model.py`, run the whole suite, then the smoke copy.
5. To run only some families, `RUN_MODELS=cfc,<name>` filters the run list.
   A filtered run is a partial DM family: give each batch its own `RUN_ID`,
   and with it `RUN_KIND` set to the config's `experiment.run_kind`
   (`main.py` refuses a `RUN_ID` without it).

## Done

Summarise what ran, where the output files are, and what the numbers do and
do not show. Say which runs were smoke runs. Do not commit unless asked.
