# Validity audit — existing forecasting pipeline (#0/#1)

**Date:** 2026-06-24
**Trigger:** Run #2 CEO review + Codex outside voice flagged macro release-timing leakage;
user asked for a systematic sweep before committing any re-run.
**Scope:** every leakage / timing / comparability vector in the existing pipeline
(`grid-2026-06-11`, AAPL/SPY/JPM daily log-return forecasting).

## How to read this

Each finding is classified on two axes:

- **Direction:** does the issue *threaten the headline* (AR(1) beats every NN), or is it
  *NN-favorable / cosmetic*? The pipeline's central asymmetry: **leakage that flows to the
  neural-net exogenous inputs is conservative** — it helps the models that lost, so AR(1)
  winning anyway makes the result *stronger*, not weaker. Only leakage that unfairly favors
  the univariate AR(1) winner, or unfairly handicaps the NNs, can threaten the headline.
- **Remedy:** *writeup-fixable* (a methods/limitations sentence) vs *needs-resnapshot*
  (would require re-fetching data + re-running the grid, breaking the pinned snapshot).

## Verdict

**No re-run is indicated.** The scored evaluation is leak-free on every axis that could
affect the result. The one candidate that could have threatened the headline — NNs denied
the predictor AR(1) uses — was already closed in run #1. Every remaining issue is
NN-favorable or cosmetic, i.e. a documented caveat, not a flaw.

## Headline-relevant axes — all GREEN

| # | Check | Finding | Evidence | Status |
|---|-------|---------|----------|--------|
| 1 | **NN input parity** (the one dangerous candidate) | NNs receive lagged close `log_return` — the same predictor AR(1) uses. Was added in run #1. | `models/lstm_model.py:261` ("past log_return as a feature (run #1)") | GREEN — parity restored |
| 2 | **Scored-eval scaler/model fit scope** | Fit is train-only: `fit(X_train)` → `predict(X_train, X_test)`; scaler fit on `X_train`. The full-series refit is only in the unscored `future_forecast` ("predict tomorrow") path, never tested against the eval window. | `main.py:557-560`, `lstm_model.py:322`; refit `lstm_model.py:650` is future-only | GREEN — no leak in the numbers |
| 3 | **Forward-shift lookahead** | No negative shift (`shift(-n)`) anywhere in feature construction. Nothing pulls future values into the present. | grep over `features/`, `models/`, `main.py` | GREEN |
| 4 | **Rolling-window features** | All `rolling(window=w)` trailing (no `center=True`); all `ewm(adjust=False)` causal. Technical indicators use past data only. NaN warmup absorbed by `warmup_discard=20`. | `features/feature_builder.py:391-476` | GREEN |
| 5 | **Financial-statement timing** | Indexed by SEC **`filed`** date (when the figure became public), not fiscal period-end; keeps the ORIGINAL filing to avoid restatement lookahead. | `data_import/internal_sec.py:122-146` | GREEN |
| 6 | **Target self-leak (SPY)** | `^GSPC` dropped when forecasting SPY (an index that nearly reconstructs the target). Per-ticker `feature_exclusions`. | `features/feature_builder.py:535-546` | GREEN |
| 7 | **Cross-run DM pairing (target)** | `assert_y_true_consistent` hard-fails if any two run_keys disagree on `y_true` at a shared `(seed,ticker,fold_id,t)`. | `analysis/merge_dm.py:52` | GREEN (target axis) |

## NN-favorable / cosmetic findings — document, do not re-run

| # | Issue | Direction | Why it can't threaten the headline | Remedy |
|---|-------|-----------|------------------------------------|--------|
| 8 | **FRED macro release-timing.** Series are stamped by reference period, not release date. `macro_lag=1` (one-day shift) does not cover the real publication lag (CPI ~6 weeks, payrolls ~1 week, GDP advance ~1 month). The NNs effectively "see" macro prints weeks before the public. | → NN exog only | Macro feeds only the NNs, which lost. An early peek at monthly/quarterly macro can only *help* the NN forecaster, so AR(1) winning is conservative. | Writeup. See severity note below. |
| 9 | **FRED revisions / vintage.** Default FRED series return the latest *revised* value; historical rows can reflect revisions made later. ALFRED vintages would fix both #8 and #9. | → NN exog only | Same asymmetry as #8. | Writeup |
| 10 | **Bidirectional interpolation of low-frequency macro.** `interpolate(method="time").ffill().bfill()` smears the next release backward across intervening days, and `bfill` fills leading NaNs from the first future observation. | → NN exog only | Same asymmetry; affects only NN macro inputs. | Writeup |
| 11 | **`bfill` of index history before first observation.** `align_to_price(..., allow_bfill_earlier=True)` backfills dates earlier than the first index print. | → NN exog, pre-window | Only touches dates before the first observation, which precede `initial_window=750` + `warmup_discard=20`; outside the scored eval window. | Cosmetic |

## Severity of the FRED macro leak (#8) — calibrated

**Is it real lookahead?** Yes. The model is fed, e.g., January CPI on Jan 2, when it was not
public until ~Feb 13. Strictly, the NN exogenous block is **not point-in-time clean**, and the
thesis cannot claim it is.

**Does it threaten any conclusion? No.** Two independent reasons:
1. **Direction.** Macro feeds only the neural nets. Any informational advantage from an early
   macro peek accrues to the models that *lost*. A handicap removed from the loser makes the
   winner's margin conservative, never inflated. The univariate AR(1) winner never touches macro.
2. **Practical magnitude.** The exploitable daily-return signal from a few-weeks-early *monthly/
   quarterly* macro print is small: between releases the value is constant/interpolated, and the
   models forecast 1-step daily returns rather than trading the release-day surprise. So even the
   conservative benefit to the NNs is likely minor.

**What it costs in the writeup:** one honest limitations paragraph — "FRED inputs are aligned by
reference period with a one-day operational lag, not by publication date or real-time vintage;
this is a (conservative) lookahead confined to the neural exogenous inputs, which were the
under-performing models; the univariate benchmark is unaffected." Fixing it properly (ALFRED
vintages) would break the pinned `grid-2026-06-11` snapshot and force a full re-run — not
justified by a hole that strengthens the result.

## Cross-run comparability caveat (for run #2, not a current defect)

`assert_y_true_consistent` covers the *target* axis but is necessary, not sufficient, for full
DM pairing (Codex #3/#8/#19): differenced/stationarized exog changes the *effective sample*
(NaNs, fold-valid rows). If run #2 adds a stationarized-exog arm, add per-feature-group, per-fold
row accounting and treat raw-vs-stationarized as a **separate** comparison, not pooled into the
main DM family. Not a flaw in #0/#1.

## What this means for run #2

- The existing science is sound; the trimmed run #2 (MA(1)/ARMA(1,1) + safeguards + param table)
  builds on a clean base.
- The validity work is now mostly **writeup**: limitations paragraphs for #8-#11, Holm family
  scope, headline wording, economic-significance note. No compute required for any of it.
- The only place new leakage risk enters is the (now-cut) SARIMAX-with-X arm and any
  stationarized-exog comparison — both covered by the run #2 plan's anti-leak test and the
  pairing caveat above.
