# Grid results analysis — AAPL (writeup material)

**Run group:** `grid-2026-06-11` · **Source:** Supabase (`experiment_results`, `experiment_summaries`)
**Setup:** expanding-window walk-forward, `initial_window=750`, fold width `horizon=21` days,
85 folds/seed, 5 seeds (42–46), 50 epochs. Target = daily **log-returns**.
**Snapshot date:** 2026-06-13. AAPL is the only ticker fully complete at time of writing
(SPY, JPM still running; ETA ~Jun 17–18). Recompute per-ticker once the grid finishes.

> **Important eval detail.** Within each 21-day fold the models produce *rolling
> **1-step-ahead** daily forecasts* (see `src/validation/walk_forward.py` docstring).
> `horizon=21` is the **test-fold width, not the forecast horizon**. Every metric below is a
> 1-day-ahead error. There is currently no multi-step (day-1-vs-day-21) error stored — see
> "Open data gap" at the end.

> **Methods note — fixed training budget.** Each neural model is **re-trained from scratch on
> every fold** (`src/main.py:688`, `fit = fit_fn(train_idx)`) — required for an expanding-window
> walk-forward, since the training window grows each fold. Training uses a **fixed budget of 50
> epochs** with **no early stopping and no validation-based epoch selection** (the 50 comes
> straight from `cfg["models"]["*"].epochs`, not a tuned value). This is a deliberate fairness
> choice: every model gets the identical a-priori budget rather than a separately tuned one, so
> the comparison is not confounded by per-model stopping heuristics. Cost: 85 folds × 50 epochs =
> 4,250 epochs per NN, per seed, per ticker (~318k epochs for the full 5-model × 5-seed × 3-ticker
> grid; ~6 days wall-clock). The classical models (AR(1), Naive) have no epoch concept.

## 1. Point-forecast accuracy (mean over 5 seeds × 85 folds)

| Model | n | RMSE | MAE | MASE | Dir. Acc |
|---|---:|---:|---:|---:|---:|
| **SARIMAX = AR(1)** | 425 | **0.01729** | **0.01312** | **0.7405** | 0.4655 |
| LTC | 850 | 0.01886 | 0.01469 | 0.8285 | 0.4766 |
| LSTM | 850 | 0.01897 | 0.01472 | 0.8304 | 0.4804 |
| NeuralODE | 850 | 0.01902 | 0.01479 | 0.8358 | **0.4881** |
| CfC | 850 | 0.01903 | 0.01479 | 0.8346 | 0.4770 |
| CTRNN | 850 | 0.02153 | 0.01691 | 0.9527 | 0.4881 |
| Naive | 425 | 0.02434 | 0.01925 | 1.0893 | 0.4527 |

Notes:
- **MASE < 1** for all but CTRNN/Naive → they beat the naive baseline on scaled error.
- **Directional accuracy ≈ 47–49% across the board** — essentially a coin flip. "Winning"
  here means smaller-magnitude error, *not* better sign prediction.
- **Drop sMAPE from the writeup**: it returns 150–177% and is meaningless because log-returns
  cross zero (denominator blows up).

## 2. Diebold–Mariano significance tests (AAPL, aggregated over 5 seeds)

Corrected p-values; `sig/5` = number of seeds significant after multiple-comparison correction.
DM sign convention: positive ⇒ model_a worse (so the "better" model is named).

| Pair (A vs B) | mean DM | median p_corr | sig/5 | Better |
|---|---:|---:|:---:|---|
| naive vs sarimax | 8.47 | 0.0 | 5/5 | SARIMAX |
| naive vs lstm | 6.51 | 3e-09 | 5/5 | LSTM |
| naive vs ltc | 6.30 | 8e-09 | 5/5 | LTC |
| naive vs cfc | 7.32 | 3e-12 | 5/5 | CfC |
| naive vs ctrnn | 6.50 | 8e-10 | 5/5 | CTRNN |
| naive vs neural_ode | 6.45 | 2e-09 | 5/5 | NeuralODE |
| sarimax vs lstm | −6.67 | 7e-10 | 5/5 | **SARIMAX** |
| sarimax vs ltc | −8.53 | 0.0 | 5/5 | **SARIMAX** |
| sarimax vs cfc | −6.42 | 5e-12 | 5/5 | **SARIMAX** |
| sarimax vs ctrnn | −6.65 | 2e-09 | 5/5 | **SARIMAX** |
| sarimax vs neural_ode | −7.06 | 1e-10 | 5/5 | **SARIMAX** |
| lstm vs ltc | 0.31 | 1.00 | 0/5 | tie |
| lstm vs cfc | 2.98 | 0.069 | 2/5 | CfC (weak) |
| lstm vs ctrnn | 0.14 | 1.00 | 0/5 | tie |
| lstm vs neural_ode | 0.28 | 1.00 | 0/5 | tie |
| ltc vs cfc | 3.20 | 0.108 | 2/5 | CfC (weak) |
| ltc vs ctrnn | −0.16 | 1.00 | 0/5 | tie |
| ltc vs neural_ode | 0.14 | 0.79 | 0/5 | tie |
| cfc vs ctrnn | −2.82 | 0.054 | 2/5 | CfC (weak) |
| cfc vs neural_ode | −2.51 | 0.106 | 2/5 | CfC (weak) |
| ctrnn vs neural_ode | 0.19 | 0.89 | 0/5 | tie |

**Reading:**
1. Every model beats Naive decisively (6/6 pairs, 5/5 seeds) — sanity check passed.
2. **AR(1) significantly beats *every* neural net in 5/5 seeds**, largest margin vs LTC.
3. The neural nets are **statistically indistinguishable from each other** (LTC≈LSTM≈NeuralODE≈CTRNN,
   0/5). Only CfC shows faint signal vs the others (2/5, above 0.05 corrected). The raw-mean
   "LTC is best NN" edge **does not survive significance testing** — state this explicitly.

## 3. Parameter counts (AAPL, n_features = 211, seq_len = 20)

Trainable parameters, counted by instantiating each network at the AAPL input dimension.

| Model | Architecture (grid config) | Params | × AR(1) |
|---|---|---:|---:|
| **SARIMAX = AR(1)** | order (1,0,0), no const, no exog | **1** (+1 σ²) | 1× |
| Naive | last-value carry-forward | 0 | — |
| CTRNN | hidden 32, dt=1 | 7,873 | ~7,900× |
| CfC | AutoNCP units 16, sparsity 0.5 | 8,372 | ~8,400× |
| LTC | AutoNCP units 16, sparsity 0.5 | 15,000 | 15,000× |
| NeuralODE | GRU enc 32 → latent 16 → ODE MLP 32 | 25,137 | ~25,000× |
| LSTM | 2 layers, hidden 64, dropout 0.1 | 104,257 | ~104,000× |

Notes:
- Input-dependent counts (LSTM, CTRNN, NeuralODE GRU encoder, LTC/CfC sensory synapses) scale
  with `n_features`; SPY/JPM rows will differ slightly (SPY drops ^GSPC → fewer features).
  AR(1) stays at 1 regardless.
- LTC (15k) > CfC (8.4k) despite identical `units=16` — LTC's ODE solver adds per-synapse
  parameters (gleak, vleak, cm, σ, μ, erev).
- σ² counted as a 2nd AR(1) param (statsmodels estimates it). Report as "one autoregressive
  coefficient" in prose, footnote the variance.

## 4. Headline for the thesis

A **single-parameter AR(1)** significantly outperforms LTC, CfC, LSTM, CTRNN and NeuralODE
(each ingesting 211 engineered features through 8k–104k parameters) on AAPL daily log-returns,
in 5/5 seeds. This is the **efficient-market / parsimony result**: daily log-returns are
near-unpredictable; the only exploitable structure is weak lag-1 linear autocorrelation, and
the extra capacity of the deep models fits noise rather than signal. Capacity does not track
performance (104k-param LSTM ties 8k-param CTRNN). The LNN family neither beats nor loses to
the LSTM baseline — a clean, citable **negative result**.

## 5. Limitation / scope note — forecast horizon

> **Limitation (writeup sentence).** This evaluation is strictly **one-step-ahead**: at each
> step the model forecasts the next day's log-return conditioned on all realized values up to
> the previous day, and the test-fold width of 21 days is a backtesting block size, not a
> forecast horizon. Multi-step (k-day-ahead) forecast-error decay is therefore out of scope.
> This is a deliberate design choice rather than a shortcoming: one-step-ahead prediction is the
> realistic and standard target for daily log-returns, whereas blind 21-day-ahead point
> forecasting of daily returns is not a meaningfully forecastable quantity (returns are
> near-unpredictable even one step out, and a sensible 21-day forecast collapses to ~0).

### Why the question is not recoverable from stored data

The natural follow-up ("does accuracy decay from a 1-day to a 21-day-ahead forecast?") **cannot
be answered from the current data**, for two reasons:

1. **The walk-forward eval is rolling 1-step-ahead.** `horizon=21` is the test-fold width, not a
   forecast horizon. All 6,039 `experiment_results` rows are 1-day-ahead errors — there is no
   per-step horizon dimension stored.
2. **The only per-step table (`experiment_future_forecasts`, `horizon_step` 1..10) is unusable
   for error analysis:** AAPL only, from a one-off run on 2026-06-08 (not the grid), and
   `actual` is NULL for all 440 rows — pure out-of-sample future forecasts with no ground truth,
   so no per-step error can be computed.

**To answer the question, a code change is required** (not in scope of the running grid):
emit a genuine multi-step (1..21) forecast per fold, align each step against its realized
actual, and store per-`horizon_step` error. Then degradation can be plotted/tested per model.
A likely outcome to anticipate: AR(1) on log-returns mean-reverts to ~0 within a couple of
steps, so its multi-step forecast collapses to "predict 0" quickly — the deep models *might*
look relatively better at longer horizons, which would be the interesting contrast.
