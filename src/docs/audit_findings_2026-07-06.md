# Codebase audit findings — 2026-07-06

Full read of the first-party codebase (~9.5k lines: `src/main.py`, models, features,
cleaning, evaluation, loggers, analysis, scripts, configs) plus read-only verification
against the Supabase input snapshots — the exact parquet frames the models trained on
(`experiment-snapshots` bucket, written by `main._resolve_df_finals`). Coverage of the
snapshot sweep: **all 3 groups (`grid-2026-06-11`, `run2a-20260625`, `run2b-20260627`)
× all 3 tickers × every stored condition** (run #0 stores raw+transformed; run #2a/2b
store transformed only, correctly — their run lists contain no raw arms), including a
`validate_feature_matrix` pass over each frame. Test suite at audit time: **300 passed**.

**Status: findings only — nothing has been fixed.** Re-running the grid before the
2026-07-28 hand-in is not feasible, so the intended consumer of this document is the
thesis **Limitations** section (suggested wording per finding below). Items marked
`[CONFIRMED]` were verified against the actual stored snapshots, not just read from code.

**Headline impact: none of these flip the conclusions.** The AR(1)/SARIMAX headline
baseline uses no exogenous features, and within each run every model saw byte-identical
inputs, so all DM comparisons remain internally fair. The findings weaken *secondary*
claims (what information the neural nets actually received, and run #2b's premise).

---

## A. Data-validity findings (confirmed against snapshots)

### A1. SEC "quarterly" income/cash-flow features mix 3/6/9/12-month durations `[CONFIRMED]`
**Where:** `src/data_import/internal_sec.py::extract_series`.
**What:** XBRL companyfacts records are taken without filtering by fact duration
(`end − start`). Income-statement and cash-flow concepts are reported at 3-month,
6-month-YTD, 9-month-YTD and 12-month durations in the same payload; the code keeps
whichever record was filed first per period-end.
**Evidence (AAPL, all snapshots):** `income_q_total_revenue` reports in sequence:
81.43 bn (Q3 FY21, 3-mo) → **365.82 bn (FY2021 annual)** → 123.94 bn (Q1) →
**221.22 bn (6-mo YTD)** → **304.18 bn (9-mo YTD)** → **394.33 bn (FY22 annual)** → …
`income_q_net_income` shows the same sawtooth (21.7 → 94.7 → 34.6 → 25.0 → 79.1 → 99.8 bn).
**Evidence (JPM — worse):** `income_q_total_revenue` alternates between the FY *annual*
total and **exactly 0.00**: 0 → 121.65 bn (FY) → 0 → 128.69 bn → 0 → 158.10 bn → 0 →
177.56 bn → 0 → 182.45 bn → 0. JPM (a bank) reports the chosen revenue concept only in
10-K annual durations, and interim filings carry literal zero-valued facts, so the JPM
"quarterly revenue" feature is an annual/zero square wave in every run.
**Affected:** all runs (#0, #1, #2a, #2b); all `income_q_*` / `cashflow_q_*` level columns
and their derived `_pctchg`/`_logdiff(_ffill)` columns. Balance-sheet (instant/point-in-time)
facts are unaffected.
**Impact:** the fundamental *flow* features fed to the neural models are a cumulative-YTD
sawtooth, not quarterly figures; derived growth rates are noise. Identical for every model
in every run, so relative comparisons stand.
**Code fix (future runs):** in `extract_series`, for duration-type concepts keep only
records with `start` present and `end − start` ≈ 80–100 days.
**Suggested limitation:** *"Fundamental flow features (income statement, cash-flow) were
extracted from SEC companyfacts without filtering by fact duration and therefore mix
quarterly and cumulative year-to-date figures. All models received the identical
(mis-scaled) series, so relative model comparisons are unaffected, but these features
should not be interpreted as clean quarterly fundamentals."*

### A2. All FRED macro features frozen after ~2025-11-10 while prices run to 2026-06 `[CONFIRMED]`
**Where:** `src/config.yaml` → `data.external.fred.start/end: "2015-11-07"/"2025-11-07"`
(hardcoded when that was the current date), while the Yahoo price `period: "10y"` tracks
the fetch date.
**Evidence:** in `run2a-20260625/AAPL/input_transformed.parquet` (prices end 2026-06-25),
the last value-change dates of all 37 macro columns are between 2025-09-02 and 2025-11-10;
same pattern in `grid-2026-06-11` (prices end 2026-06-10, 10-y yield last changes 2025-11-03).
Everything after is forward-fill.
**Affected:** all runs. In run #2b the stationarized (differenced) macro columns are
exactly **zero** over the frozen tail.
**Impact:** the final ~7 months (~150 trading days ≈ the last ~7 of 86 walk-forward folds)
were trained and evaluated with stale macro inputs. Headline unaffected (AR(1) is
exog-free), but it weakens "the neural nets received macro information" and run #2b's
premise for the tail folds. The data-quality gate cannot detect this (columns still vary
over the full sample).
**Code fix (future runs):** derive `fred.end` from the run date; optionally add a DQ-gate
check that each macro column's last change is within N days of the price tail.
**Suggested limitation:** *"The FRED download window was fixed at config-writing time
(ending 2025-11-07), while equity prices extend to the run date; the final ≈7 months of
macro features are therefore stale forward-fills (constant, and zero after differencing
in run #2b). This affects all models identically."*

### A3. Run #2b missed one macro series — it stayed a raw level `[CONFIRMED]`
**Where:** key-spelling mismatch between `src/config.yaml` / `src/features/stationarize.py`
(`us_real_manufacturing_and_trade_sales_inflation_adjusted`) and the sanitized data key
produced by `FeatureBuilder.sanitize("U.S. Real Manufacturing & Trade Sales
(Inflation-Adjusted)")` → `us_real_manufacturing_trade_sales_inflation_adjusted`
(the `&` collapses into the underscore run; no "and" is inserted).
**What happens:** `FeatureBuilder.build_macro_features` looks up feature metadata by the
sanitized *data* key, misses, and falls back to `transform: "none"`. The fail-fast in
`stationarize_exog_cfg` compares *config keys ↔ table keys* (both spelled "and"), so it
structurally cannot catch a config↔data mismatch.
**Evidence:** in `run2b-20260627/AAPL/input_transformed.parquet` that column holds raw
levels 1.18M–1.57M while every other macro column is a log-return/diff on the order of
0.01 (e.g. `us_m2_money_stock` max |x| = 0.062).
**Affected:** run #2b materially (1 of 37 macro series unstationarized). In runs
#0/#1/#2a the miss is harmless (the intended transform was `none` anyway), but the
series' declared metadata (units/description) has never been applied in any run.
**Doc impact:** `src/docs/thesis_methodology.md:202` — "forces **every exogenous
feature** through the fixed per-feature transform table" — is factually false as written.
**Code fix (future runs):** rename the config/table key to the sanitized data spelling
and add a build-time assertion that every declared `features.macro`/`features.index` key
matches an actual data column (kills the whole bug class).
**Suggested correction:** soften the methodology sentence to "every exogenous feature
except one (a key-sanitization mismatch left `us_real_manufacturing_&_trade_sales` at its
raw level; 1 of 37 macro series)" — or fix and regenerate the config for the record.

### A4. Fundamentals cover only the last ~5 of 10 years `[CONFIRMED]`
**Where:** `src/data_import/internal_sec.py::build_statement` (and `wide`) keep
`df.iloc[-20:]` ("last ~20 quarters ≈ 5 years"), predating the scenario widening to 10y.
**Evidence:** fundamental columns in the snapshots are exactly 50% non-NaN; earliest
fundamental observation 2021-07-28 in a sample starting 2016-09.
**Impact:** the first half of the sample has no fundamentals; the models' internal
sanitizer mean-fills those rows (train-window mean → constant placeholder), and the
derived `_pctchg/_logdiff` columns are `fillna(0.0)` there.
**Code fix (future runs):** `iloc[-44:]` (or drive from the scenario period).
**Suggested limitation:** one sentence noting fundamental coverage begins ~2021 and
earlier rows carry constant fill values, identically for all models.

### A4b. Run #0/#1 snapshots fail today's DQ gate with 12 errors each (known bug, now snapshot-verified) `[CONFIRMED]`
Running `validate_feature_matrix` over the pinned `grid-2026-06-11` frames (the inputs
runs #0 **and** #1 trained on, via `INPUT_GROUP_ID`) reports 12 errors per ticker/condition:
4 all-NaN macro columns (`us_consumer_price_index`, `us_core_cpi`, `us_unemployment_rate`,
`us_labor_force_participation`) and 8 daily FRED series downsampled to ~monthly steps
(the 5 treasury yields, 10y–2y spread, WTI, GVZ — value changes on only ~3% of days).
This is the already-documented FRED freq-inference pitfall (fixed before run #2; the
run2a/2b snapshots validate clean at 0 errors), but it is now confirmed **in the stored
training data itself**, not just in code review: runs #0 and #1 trained with 4 dead and
8 degraded macro features. Already on the limitations checklist; cite the snapshot
validation as the evidence artifact.

### A5. Minor lookahead vectors (low priority; mostly latent)
- `src/data_cleaning/cleaner.py::clean_fred_series` uses `interpolate(method="time")`:
  interior gaps are filled using the *next* (future) observation — a small interior-gap
  lookahead, dominated by the already-documented release-lag issue. Footnote-level.
- `src/features/feature_builder.py::align_to_price(..., allow_bfill_earlier=True)`
  backfills **index** columns at the head of the sample with the first *future*
  observation. Currently inactive (all 8 indices/ETFs predate the 2016 sample start),
  but a latent leak if an instrument with later inception is ever added.

---

## B. Statistics / evaluation edge cases

### B1. `dm_test_stat` has no length-equality guard; secondary DM can misalign
`src/evaluation/evaluator.py::dm_test_stat` never asserts `len(e1) == len(e2)`.
The secondary C1-vs-raw DM block in `src/main.py` (≈ lines 1650–1660) concatenates each
model's **own** successful fold set; with the 90% fold-success threshold, the transformed
and raw variants of a model can retain *different* fold subsets (e.g. 78/86 vs 86/86),
producing mismatched arrays → numpy raises → the run crashes after fold persistence but
**before the summary row is written**. Not hit in run #2 (no raw arms); latent landmine
for any config with `condition: raw` rows. Fix: assert equal lengths in `dm_test_stat`
and intersect fold ids for secondary pairs.

### B2. Metrics table mixes fold bases
`Evaluator.compute()` averages primary models over the **aligned** fold set but
secondary (raw-arm) models over **all** their folds. Fine, but if raw-arm numbers are
ever quoted next to primary ones in the thesis, say so in methods.

### B3. Zero-return days scored inconsistently across DA computations
`main._fold_metrics` / `evaluator.directional_accuracy` score `sign(y_true)=0` days as
incorrect (unless the prediction is exactly 0), while
`scripts/build_directional_skill_figure.py` masks `y_true == 0` out before computing DA.
Differences are tiny but the two DA numbers are not strictly the same statistic.

---

## C. Code hygiene (examiner-visible)

### C1. Live `[DEBUG]` prints every training batch
`src/models/cfc_model.py:62,87` and `src/models/ltc_model.py:62,87` — the
`print(f"[DEBUG][train] x:... y:... y_hat:...")` lines are **active** (the comment says
"uncomment to inspect once"). They execute on every batch of every epoch of every fold —
the source of the multi-million-line VM logs. Similarly `SARIMAModel.fit` prints the full
`model_fit.summary()` per fold and `SARIMAModel/VARMAXModel.predict` print six metric
lines per fold.

### C2. Dead legacy code with hardcoded personal paths in `src/main.py`
- `get_features` (main.py:201) contains an **active**
  `df.to_excel("/Users/juliusschmidt/Desktop/BachelorThesisFeatures.xlsx")`.
- `get_compliance_check` (main.py:399) writes two more Desktop files.
- `SARIMAModel.results_vis` / `VARMAXModel.results_vis` save PNGs to Desktop paths.
- ~500 lines of superseded single-shot orchestration (`get_sarimax`, `get_varmax`,
  `get_lstm/ltc/cfc`, the `build_*_grid` builders) remain in main.py, reachable only
  from a commented-out block. Recommend deleting (git history preserves them).

### C3. `DataCleaner.clean_calendar` is broken
`src/data_cleaning/cleaner.py:298` calls `pd.to_datetime()` with **no argument** —
always raises TypeError, silently caught, returns the raw value. Currently unused.

### C4. Smoke-test stub logger is missing half the logger API
`src/smoke_test.py::_StubLogger` lacks `upload_snapshot`, `log_results_batch`,
`summary_exists`, `log_summary`. `main._resolve_df_finals` calls
`logger.upload_snapshot` **unguarded**, so `SMOKE_STUB_LOGGER=1` (or the automatic
Supabase-unreachable fallback) crashes with AttributeError instead of smoking offline.

### C5. Legacy `train_test_split` mean-fills with full-sample means (leakage pattern, dead path)
The neural models' `train_test_split` runs `sanitize_dataframe` (mean-fill) on the FULL
frame **before** splitting, so test-period means would leak into train fills. Only the
dead single-shot path uses it; the walk-forward path fills with train-only means.
Delete together with C2 so it cannot be resurrected.

### C6. Cosmetics
- `main.get_cfc`'s exception handler prints "**LTC** combo failed" (copy-paste).
- `CfCModel`/`LTCModel` never initialize `self.scaler_X/scaler_y` in `__init__`, so
  predict-before-fit raises AttributeError instead of the intended ValueError
  (CTRNN/NeuralODE/LSTM do it correctly).
- `scripts/build_results_figures.py:348`: predicate
  `m.startswith("sarimax") or m.startswith("naive") and False` — the `and False` is a
  confusing no-op.
- `models.VARMAX.endog_cols` in config references stale column names
  (e.g. `income_q_gross_profit_pctchg`, `dji`); VARMAX is not in any run list and its
  own code comments acknowledge it would crash under walk-forward with univariate endog.

### C7. Non-stationary levels in the "transformed" condition (design fact, worth one sentence)
The transformed-condition X still carries 37 raw macro levels (the
`stationarize_reporting_series` helper *appends* derivatives, keeping the levels),
55 raw fundamental levels, ~10 price-level technical indicators (SMA spans 24 → 299 over
the sample), and raw volume — under a z-scaler fit on the training window only, so late
test folds see feature values far outside the training range. Run #2b stationarizes
macro/index only; technicals and fundamental levels are never stationarized in either
arm. The existing Codex correction in the methodology doc covers the 2a-vs-2b framing
but not these columns.

---

## D. Suggested thesis-text changes (summary)

1. **Correct** `thesis_methodology.md` §B4: "every exogenous feature" → all but one (A3).
2. **Add limitations:** SEC duration mixing (A1), FRED end-date staleness (A2),
   fundamental coverage starting ~2021 with constant fill before (A4), interior-gap
   interpolation micro-lookahead (A5, footnote), non-stationary level columns in the
   transformed condition (C7, one sentence).
3. Existing limitations (release-lag lookahead, FRED freq bug, 2b confound, seed
   order-dependence) already cover the rest of the known issues; A1–A4 are additive.

## E. If any code fixes are made later (post hand-in)

Priority order: C1 (debug prints) → A3 key rename + declared-key-matches-data assertion
→ B1 length guard → A2 date derivation + DQ staleness check → C3/C4/C6 → C2/C5 dead-code
removal → A1 duration filter + A4 window bump. Each is small and none invalidates the
stored results; verify with `venv/bin/python -m pytest tests/ -q` (300 green at audit
time) and re-run `scripts/gen_run2b_config.py` after any config edit.
