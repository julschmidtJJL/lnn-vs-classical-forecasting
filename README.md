# Liquid Neural Networks vs. Traditional Models

Experiment code for my bachelor's thesis at Leuphana University Lüneburg (B.Sc. Business Informatics, 2026; thesis graded 1.0):
*Liquid Neural Networks vs. Traditional Models: A Comparative Study in Financial Time Series Forecasting.*

**[Read the thesis (PDF, 89 pages)](https://github.com/julschmidtJJL/lnn-vs-classical-forecasting/releases/download/thesis-2026/Schmidt-2026-Liquid-Neural-Networks-vs-Traditional-Models.pdf)**

**Question.** Do liquid neural networks forecast daily financial returns better than classical models?

**Answer.** No. Across 11 liquid variants and 8 baselines on three US assets, **0 of 120 corrected comparisons** show a liquid model significantly beating a traditional baseline.

## Findings

- **No significant edge.** No liquid variant significantly outperforms the traditional baselines on any asset (Diebold–Mariano with small-sample correction, Holm-adjusted).
- **Best model is not a real win.** The best variant, a pure-gated, sparsely wired CfC, topped the point metrics on all three assets. That lead is not significant, and its directional edge is fully explained by market drift.
- **Liquid models do differ from other neural models.** Within the neural field, liquid models separate cleanly and significantly. The measured expressivity (trajectory length) orders the neural field but does not translate into skill against the traditional bar.

The limit is the data, not the model: daily returns carry too little signal for extra capacity to pay off.

![MAE by model and asset, mean ± std over 5 seeds. The best CfC variant ties the ARMA baselines (dashed line).](src/docs/figures/fig1_mae_ranking_run2a.png)

*Forecast error (MAE) by model and asset. The best liquid variant (`cfc_pure_ncp`) sits level with the classical ARMA baselines; the difference is not significant.*

![Directional accuracy of the best CfC vs. an always-long base rate.](src/docs/figures/figD_directional_skill.png)

*The best model's directional hit rate matches simply always predicting "up". Its apparent edge is market drift, not skill.*

## Setup

| | |
|---|---|
| Assets | AAPL, JPM, SPY, ten years of daily data |
| Target | one-step-ahead log return |
| Evaluation | expanding-window walk-forward: 750-day initial window, 21-day folds |
| Seeds | 5 (42–46); point metrics as mean ± std, significance reported per seed |
| Significance | Diebold–Mariano (Harvey–Leybourne–Newbold correction), Holm adjustment |
| Liquid models (11) | LTC and CfC variants across gating mode, wiring (fully connected / sparse NCP) and memory |
| Baselines (8) | LSTM, CT-RNN, Neural ODE, Naive, four ARMA specifications |
| Inputs | price history (yfinance), macro series (FRED, lagged one day for publication delay), SEC EDGAR fundamentals |

Every run records its provenance (git commit, config, seed, input snapshot hash), so each number in the thesis traces back to the exact code and data that produced it.

## Repository layout

```
src/
  main.py            orchestrator: data → features → walk-forward → models → DM/Holm tests
  config*.yaml       run definitions (config.yaml = main grid, config.smoke.yaml = quick check)
  data_import/       yfinance, FRED and SEC EDGAR loaders
  data_cleaning/     cleaning and stationarity handling
  features/          feature engineering
  models/            ltc, cfc, lstm, ctrnn, neural_ode, naive, sarimax (ARMA), varmax
  evaluation/        metrics, Diebold–Mariano, Holm, Friedman
  analysis/          seed aggregation, expressivity / trajectory-length analysis
  provenance.py      run identity and input snapshots
tests/               pytest suite (33 modules)
scripts/             figure builders, data-quality and stationarity checks
src/docs/            methodology and results notes, figures and their data
docs/                validity audit
```

## Running it

Requires Python 3.11.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # add your own FRED API key (free at fred.stlouisfed.org)
pytest -q tests             # test suite
cd src && python main.py config.smoke.yaml   # quick end-to-end run: AAPL, ~3 folds
```

The full grid (`config.yaml`: 3 assets × 19 models × 5 seeds) takes many hours on CPU. A `Dockerfile` and a cloud run guide (`CLOUD_RUN.md`) are included for running it on a rented Linux VM.

Results are logged to Supabase, so a full run expects Supabase credentials in `.env`. If a Supabase write fails, the row is kept in a local JSONL file under `results/logger_fallback/` and the run continues.

## Limitation worth knowing

The ARMA baselines were fitted without an intercept (the statsmodels SARIMAX default). A refit with an intercept leaves the null result unchanged; the thesis discloses this.

## About this repository

This is a clean snapshot of my private development repository, published without its history. Run provenance in the results (git commit hashes) refers to that private history. The papers I worked from are cited in the thesis, not redistributed here. `vendor/` holds the reference implementations of LTC and CfC by their original authors, under their Apache 2.0 licenses.

## Author

Julius M. Schmidt, Hamburg · [juliusmschmidt.com](https://www.juliusmschmidt.com)
