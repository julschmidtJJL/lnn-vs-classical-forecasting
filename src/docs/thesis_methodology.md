# Methodology — bullet skeleton (for prose expansion)

*★ = bullet added 2026-07-06 (post-hoc input-data audit, `src/docs/audit_findings_2026-07-06.md`).*

> Draft skeleton written from the code + configs, to be turned into full prose by the author.
> **Structure:** Part A = the shared experimental apparatus (described once). Part B = the
> experiment sequence, where each run states only what it changed vs. Part A. Part C = the
> expressivity study. Part D = limitations.
>
> **Marker legend:**
> - **✅ (code-confirmed / RESOLVED)** — traced directly to source or git/Supabase provenance.
> - **⚠️ CORRECTION (Codex)** — a factual error caught in an independent Codex fact-check pass; the
>   corrected statement (with file:line evidence) is what to write. All Codex findings are applied.
> - **⚠️ ... TO DISCLOSE / DEFERRED / SCOPE** — not errors: honest limitations and scope notes to
>   fold into the Limitations section (Part D). Nothing left to verify against code.
>
> There are **no open ⚠️ VERIFY items** — every inference has been resolved against the code,
> git history, or the Supabase provenance record.

This study ran **three forecasting experiment runs (#0 → #1 → #2)** plus a separate **latent
trajectory-length expressivity study**. Run #2 (arms 2a and 2b) is the definitive forecasting
experiment; #0 and #1 are precursor ablations that close specific validity concerns. All three
forecasting runs share the apparatus in Part A and reuse the same frozen input snapshot
(`INPUT_GROUP_ID = grid-2026-06-11`).

---

# PART A — Common experimental setup (shared by all forecasting runs)

## A1. Research question & design
- Empirical forecasting study: do continuous-time / "liquid" neural networks (**LTC**, **CfC**) and
  other neural sequence models beat classical time-series baselines at **1-day-ahead log-return
  prediction** on real equity data?
- Three assets for cross-sectional contrast: **AAPL** (large-cap tech), **SPY** (broad-market index
  ETF), **JPM** (large-cap financials).
- Target: daily **log-return** of the close (continuously compounded, `log_return`).
- Prediction task: rolling **1-step-ahead** daily forecast under walk-forward evaluation.
- Overall orientation: a **negative result** — classical AR(1)/SARIMAX is not beaten by the neural
  models. The expressivity study (Part C) then probes the **motivation** for expecting a liquid-network
  win — is the theoretical trajectory-length advantage even real on financial data, and does it predict
  forecast skill? — **not** the *cause* of the loss. Keep this distinction throughout: the thesis
  explains why the *expectation* of an LTC advantage was weaker than advertised, it does not claim
  expressivity *causes* the underperformance.

## A2. Data sources
- **Price/volume**: Yahoo Finance OHLCV, 10-year daily history.
- **Firm fundamentals**: SEC EDGAR quarterly filings (income statement, balance sheet, cash flow,
  EPS), **indexed by filing date, not period-end** ✅ — no look-ahead on fundamentals.
  - **⚠️ LIMITATION TO DISCLOSE — duration mixing**: XBRL facts were extracted without filtering by
    fact *duration*, so the income-statement and cash-flow "quarterly" series mix 3-month, 6/9-month
    YTD and 12-month (annual) figures (snapshot-verified: AAPL quarterly revenue alternates
    81bn → 366bn → 124bn → 221bn; JPM's revenue concept is reported only annually and alternates
    FY-total ↔ 0). Balance-sheet (point-in-time) items are unaffected. Identical for every model in
    every run → relative comparisons unaffected; the fundamental *flow* features must not be read as
    clean quarterly figures. Full evidence: `src/docs/audit_findings_2026-07-06.md` (A1). ★
  - **⚠️ LIMITATION TO DISCLOSE — coverage**: the SEC extractor keeps only the last ~20 quarters, so
    fundamentals begin ~2021 within the 10-year window (≈50% of rows NaN, mean-filled by the models'
    input sanitizer with train-window means). Audit doc A4. ★
- **Macro**: ~36 FRED series (inflation, labor market, rates & yield curve, production, housing,
  money & credit, trade, commodities).
  - **⚠️ LIMITATION TO DISCLOSE — staleness tail**: the FRED download window was fixed at
    config-writing time (`end: 2025-11-07`) while Yahoo prices track the run date, so **every macro
    column is a frozen forward-fill over the final ≈7 months** (~150 trading days ≈ the last ~7 of
    86 folds; snapshot-verified: last value-change ≤ 2025-11-10 in all run groups and tickers). In
    run #2b the differenced macro columns are exactly 0 over that tail. Symmetric across models;
    the exog-free AR(1) headline is unaffected. Audit doc A2. ★
- **Global markets**: index & ETF levels — S&P 500, Nasdaq-100, Dow Jones, DAX, EuroStoxx 50, Gold
  (GLD), MSCI World (URTH), MSCI ACWI.
- All external inputs merged onto the daily trading calendar.

## A3. Feature engineering & leakage control
- ~20 **technical indicators** engineered from price/volume (SMA, EMA, MACD, RSI, ATR, Bollinger /
  Keltner / Donchian bands, stochastic oscillator, Williams %R, ADX, OBV, CMF, ROC).
- **Target-leakage exclusion**: the SPY model drops `^GSPC` (it *is* SPY's underlying index)
  (`feature_builder.py:539-552`) ✅ → SPY has slightly fewer features than AAPL/JPM (a cross-ticker
  caveat).
- **Publication-lag shift**: all FRED macro + index columns shifted by `macro_lag = 1` trading day
  (`feature_builder.py:612-618`) ✅.
  - **⚠️ LIMITATION TO DISCLOSE**: monthly macro series are indexed to their *reference date* and
    shifted only 1 day, not by the true ~2–6 week release lag → a mild look-ahead on macro features.
    Documented-by-design and argued **conservative**: the winning AR(1) baseline uses *no* exogenous
    inputs, while the neural models receive the (slightly leaked) macro and still lose.
- **Warmup discard**: `warmup_discard = 20` rows dropped to clear rolling-window NaN warmup ✅.
  - **⚠️ CORRECTION (Codex)**: this is *not* the only row trim — `FeatureBuilder` also does an
    unconditional earlier `df.iloc[33:]` drop (`feature_builder.py:620-621`). State the total warmup
    removed, not just the 20.

## A4. Preprocessing & scaling
- Two scaling conditions are defined: `transformed` → **z-score (StandardScaler)**, `raw` →
  **min-max (MinMaxScaler)** (`config.yaml:12-14`; `zscore`→StandardScaler at
  `neural_ode_model.py:15-18`). Which condition each run uses is a per-run choice (see Part B).
- **✅ (code-confirmed) leakage-safe scaling**: within every walk-forward fold the scaler is fit on
  the **training window only** (`scaler.fit(X_train)`; e.g. `cfc_model.py:376-391`,
  `ctrnn_model.py:187-195`, `neural_ode_model.py:212-220`), then applied to the test window — it
  never sees test data. Each fold refits its own scaler (per-fold `fit_fn(train_idx)` at
  `main.py:793-796`).
- **Neural models receive their own past log-returns as an input feature** when
  `include_past_target_as_feature = true` (`config.yaml:38-45`; policy in `feature_policy.py:23-48`)
  — legitimate autoregression under the windowed 1-step-ahead eval (the input window `X[i:i+seq_len]`
  ends at *t−1* when predicting `y[i+seq_len]`, `main.py:1356-1366`), **not** look-ahead leakage. The
  classical AR(1)/SARIMAX models use the past log-return directly. *(This flag is exactly the #0→#1
  change — see Part B.)*

## A5. Model training protocol
- All neural models: **fixed 50 epochs, no early stopping**; sequence length 20; batch size 32;
  learning rate 0.001 (LSTM) / 0.005 (LTC, CfC, CTRNN, Neural ODE) (`config.yaml` models section).
- **✅ RESOLVED — all three runs used a uniform 50 epochs.** Run #0 (`grid-2026-06-11`) is pinned in
  Supabase provenance to commit `ae31f57`, which already includes the uniform-50 standardization
  (`5579894` is its ancestor). The non-uniform budget (LSTM 10, LTC 1, CfC 1) existed only in
  pre-grid development commits (`41e4bbf`) and was never used in a published grid run.
- **⚠️ CORRECTION (Codex) — gradient clipping is NOT applied to all neural models.** LTC/CfC use
  `clip_gradient = 1.0` and CTRNN/Neural ODE use `grad_clip = 1.0`, but **LSTM has no gradient
  clipping** — its loop calls `loss.backward(); optimizer.step()` with no clip step
  (`lstm_model.py:376-386`; no clip param in `config.yaml:1441-1449`).
- **Refit from scratch every fold** (no warm-start across folds) — deliberate uniform-budget fairness
  choice. **CPU-only**, deterministic RNG seeding per (ticker, seed).
- **⚠️ SEED LIMITATION TO DISCLOSE**: `_seed_everything` is called once per (ticker, seed), not per
  fit, so neural initialisations depend on run-list order. Adjudicated not-a-bug (both arms share the
  same order); disclosed as a limitation.

## A6. Evaluation — expanding-window walk-forward ✅
- **Expanding-window walk-forward** (`WalkForwardSplitter`): initial training window **750 trading
  days (~3 yr)**; each **test fold is a fixed 21-day (~1 month) block**; the training window grows by
  21 days per fold; test blocks are **non-overlapping** (≈80+ folds over 10y) (`walk_forward.py:43-50`).
- **`horizon` here = the test-fold size in days, not the forecast horizon** (`walk_forward.py:10-15`).
  Within each 21-day test block the model emits **rolling 1-step-ahead daily forecasts** (fit once
  per fold; SARIMAX `forecast(1)` then `append(..., refit=False)` at `sarimax_model.py:200-219`).
- **Metrics**: **RMSE, MAE, and directional accuracy** — reported **mean ± std across folds**
  (`evaluator.py:217-239`).
  - **⚠️ CORRECTION (Codex)**: a `mape` function exists but `Evaluator.compute()` does **not** report
    MAPE — don't claim it unless pulled from a different code path.

## A7. Multi-seed robustness ✅
- Each run's full grid is executed once per seed for **5 seeds (42–46)**.
- Point metrics reported **mean ± std across the 5 seeds**; DM significance reported **per seed as
  k-of-N significant**.
- **Frozen inputs**: seed-1 of a run group builds the feature matrix from live data and uploads a
  parquet snapshot + fingerprint; seeds 2–5 **load that exact snapshot**. All three runs pin the same
  Run #0 input group via `INPUT_GROUP_ID = grid-2026-06-11`, so every run/seed sees identical inputs.

## A8. Statistical testing ✅
- Pairwise **Diebold–Mariano** tests on the forecast-error differentials, with the
  **Harvey–Leybourne–Newbold (1997) small-sample correction** and **Newey–West HAC** variance
  (bandwidth = h−1 lags) (`evaluator.py:45-80`); `Evaluator(horizon=1)` since errors are 1-step-ahead
  (`main.py:1626-1629`).
- **Holm–Bonferroni** correction across the pairwise comparison family (`evaluator.py:83-103, 277-286`).
- **Friedman rank test** across assets (models as treatments, per-asset RMSE ranks)
  (`evaluator.py:288-358`).
  - **⚠️ CORRECTION (Codex)**: the code explicitly labels this **"descriptive evidence"**, not a
    confirmatory test (`evaluator.py:298-301`). Present it as descriptive/exploratory ranking support,
    **not** an "overall ranking test" that confirms significance.

## A9. Reproducibility & infrastructure ✅
- Fully **Dockerized**; runs on **Hetzner cloud VMs** (CPU-only; one (ticker × run) per box).
- Results logged to **Supabase**: per-fold `experiment_results`, per-seed `experiment_summaries`, and
  per-seed forecast snapshots to object storage.
- **Provenance pinning** with every run: git commit hash, config snapshot, environment manifest, and
  input data fingerprint. (This is how Run #0's epoch budget was resolved above.)
- **Data-quality gate** runs before any model sees data — hard-fails on all-NaN columns, infinities,
  or silently downsampled (frequency-mangled) series.

---

# PART B — Experiment sequence

Every run reuses Part A. The table lists only what each run changed; subsections give the detail.

| Run | RUN_ID (git commit) | Past-target feature | Scaling conditions | Model set | Purpose |
|---|---|---|---|---|---|
| **#0** | `grid-2026-06-11` (`ae31f57`) | **OFF** (`false`) | transformed **+ raw** | 7 models × conditions (13 run-rows) | Baseline; establish the negative result |
| **#1** | `run1` (`config.run1.yaml`) | **ON** (`true`) | transformed **+ raw** | same as #0 | Input-fix ablation: did the input handicap explain the loss? |
| **#2a** | `run2a-20260625` (`run_kind: run2`) | ON | transformed **only** | 19-model variant grid | Architecture/wiring sensitivity + auto-order SARIMAX |
| **#2b** | `run2b-20260627` (`run_kind: run2b`) | ON | transformed **only** | same 19-model grid | Robustness to stronger exogenous stationarization |

## B1. Run #0 — baseline (`grid-2026-06-11`)
- **Change vs. Part A**: neural models do **not** receive the past log-return
  (`include_past_target_as_feature = false`, historical `git show e5890d9:src/config.yaml:34-39`) — a
  deliberately handicapped-input baseline.
- **Model set** (7 models): naive, SARIMAX = AR(1) (`order (1,0,0)`), LSTM, LTC, CfC, CTRNN, Neural
  ODE. naive + SARIMAX run in the `transformed` condition; each of the 5 neural models runs in **both
  `transformed` (z-score) and `raw` (min-max)** conditions → 13 run-list rows (`ae31f57:src/config.yaml:49-65`).
- **Result orientation**: classical AR(1)/SARIMAX wins on all assets — the negative result the thesis
  is built around.

## B2. Run #1 — input-fix ablation (`config.run1.yaml`)
- **Change vs. #0**: a single one-variable flip — `include_past_target_as_feature = true`
  (`config.run1.yaml:44`), so neural models now receive their own past returns. Same 13-row run-list
  and same transformed+raw conditions as #0 (`config.run1.yaml:70-83`).
- **Purpose**: isolate whether #0's handicap (no past target) explained the neural loss.
- **Finding**: it did **not** — the classical baseline still won — which is why Run #2 inherits the
  fixed-input setting.

## B3. Run #2a — variant grid, base transforms (`run2a-20260625`, `run_kind: run2`)
- **Changes vs. #1**:
  1. Neural models kept at past-target ON.
  2. **Dropped the `raw` (min-max) condition** — Run #2 uses `condition: "transformed"` only
     (`config.yaml:79-106`; 0 raw rows). So the transformed-vs-raw scaling comparison was a #0/#1
     device; #2 replaces it with the exogenous-stationarization arms (2a vs 2b).
  3. **Expanded to a 19-model architecture/wiring grid**, self-contained (all baselines recomputed
     in-run, no cross-run snapshot dependency).
- **Model set (19)** ✅ (`config.yaml:79-106`):
  - **Classical (5)**: naive; AR(1) = SARIMAX(1,0,0); MA(1) = (0,0,1); ARMA(1,1) = (1,0,1);
    **auto-order SARIMAX** = per-fold AIC search over p,q ∈ {0,1,2}, d = 0, no seasonal term
    (`config.yaml:1418-1424`; `sarimax_model.py:142-149`).
  - **⚠️ NAMING NOTE — "SARIMAX" is a misnomer; every such model actually run is a plain,
    non-seasonal, exog-free ARMA on the log-return series.** The label is inherited from the
    statsmodels `SARIMAX` class, but (i) the model is fit `SARIMAX(endog=y_train, order, seasonal_order)`
    with **no `exog=` argument** — exogenous features are never passed to any classical model
    (`sarimax_model.py:94`, `:147`; classical dispatch ignores `X_df`, `main.py:1362-1364`); and
    (ii) `seasonal_order = (0,0,0,0)` in **every reported run**, so there is no seasonal (S) term.
    Differencing is `d = 0` because the target is already the log-difference (integration
    pre-applied). Confirmed against Supabase across all 13,151 SARIMAX-family `experiment_results`
    rows: the only rows with a non-zero seasonal order are 4 one-off pre-`run_id` grid diagnostics
    (June 11) that are **not** part of any reported run. **Implication for the theoretical section:**
    the baseline family is Box–Jenkins **ARMA(p,q) on stationary returns** — AR(1) headline, plus
    MA(1), ARMA(1,1), and an AIC-selected ARIMA(p,0,q) (p,q ≤ 2) — **not** seasonal ARIMA and **not**
    exogenous-regressor SARIMAX. Across all folds/seeds/tickers the AIC search overwhelmingly selects
    low-order AR — AR(2) (1,195 folds) or AR(1) (705) ≈ 74% — consistent with the near-unpredictable
    log-return headline.
  - **CfC (8)**: {default, pure, no-gate} × {sparse AutoNCP wiring (16 units), fully-connected + MLP
    backbone (16 units)}, plus a mixed-memory (LSTM-augmented) variant and a wide-backbone (128-unit)
    variant.
  - **LTC (3)**: AutoNCP wiring, fully-connected wiring, mixed-memory.
  - **Neural baselines (3)**: LSTM (hidden 64, 2 layers, dropout 0.1), CTRNN (hidden 32; Funahashi &
    Nakamura 1993), Neural ODE (latent 16, GRU encoder + latent-ODE residual, RK4 solver, step 0.25;
    Chen et al. 2018) (`neural_ode_model.py:49-98`).
- **⚠️ SOURCE-REPO DIVERGENCE TO DISCLOSE**: LTC/CfC use `AutoNCP(units=16)` — the NCP sparse-wiring
  paradigm — deliberately unlike the reference papers' fully-connected + backbone, larger-hidden-size
  configuration. Numbers are not directly comparable to the source repos; intentional design choice,
  not a replication.
- **Exogenous transforms**: the base transformed pipeline only. `exog_stationarize` is unset.

## B4. Run #2b — + stationarized exogenous features (`run2b-20260627`, `run_kind: run2b`)
- **Change vs. #2a**: identical 19-model grid, but adds `exog_stationarize = true`
  (`config.run2b_stationary.yaml:22-31`), which forces **every exogenous feature** through the fixed
  per-feature transform table `features/stationarize.py` (`main.py:1874-1886`), transforms chosen from
  ADF/KPSS testing.
  - **⚠️ CORRECTION (audit 2026-07-06) — "every" is every exogenous feature *but one*.** A
    key-sanitization mismatch (`..._and_trade_sales...` in the config/table vs the sanitized data key
    `..._trade_sales...` — the "&" in the FRED display name collapses without inserting "and") means
    `us_real_manufacturing_&_trade_sales` silently fell back to `transform: none` and **stayed a raw
    level in run #2b** (snapshot-verified: values ~1.2–1.6M vs ~0.01 for the other 36 stationarized
    macro columns). The `stationarize_exog_cfg` fail-fast compares config↔table keys, so it cannot
    catch config↔data misses. 1 of 37 macro series; the 2b ≈ 2a conclusion is unchanged, but do not
    write "every" in prose. Audit doc A3. ★
- **⚠️ CORRECTION (Codex) — do NOT frame 2a as "raw / native levels" and 2b as "the only
  stationarization".** **Both** arms use `condition: "transformed"`, so **both** already log-difference
  index columns (`compliance.py:425-443`) and append macro stationarity derivatives
  (`compliance.py:449-573`) via `create_stationarity` (`main.py:372-375`). Run 2a is already partly
  stationarized; **2b is a stronger / more uniform exogenous stationarization layered on top of 2a**,
  not a raw-vs-stationarized contrast.
- **Target**: `log_return` is **treated as stationary** — SARIMAX uses `d = 0` and the
  stationarization step leaves the internal target untouched (`stationarize.py:109-112`).
  - **⚠️ CORRECTION (Codex)**: the "ADF/KPSS-confirmed I(0)" claim is **not backed by a committed
    result artifact** — only comments. If asserted in prose, cite your own
    `scripts/stationarity_methods_check.py` output / the ADF & KPSS statistics, don't state it bare.
- **⚠️ LIMITATION TO DISCLOSE (2b confound)**: because 2b changes the exogenous representation as a
  block on top of 2a's existing transforms, it is a sensitivity probe, not a clean single-variable
  isolation.

---

# PART C — Latent trajectory-length expressivity study ("opening the black box")

> Analysis of the models' latent geometry at random init (a different object from trained forecast
> skill). Asks whether the *theoretical* liquid-network expressivity advantage (Hasani et al. 2021,
> measured as latent **trajectory length**) is real on financial data, using Hasani's own random-init
> protocol; **C6 then bridges it to forecast skill** rather than leaving the two disconnected.

## C1. Measure ✅
- **Arc length** of the latent state path as a model processes an input sequence:
  `L = Σ_k ‖z_{k+1} − z_k‖₂`, measured on the **first 2 principal components** of the latent path
  (`trajectory_length.py:5-12, 77-85`). Longer L ⇒ more expressive latent dynamics. Theoretical
  ordering: L(LTC) ≥ L(CT-RNN) ≥ L(Neural-ODE).
- **Scale-invariant normalisation**: `L_norm = arc_length(proj) / ((T−1) · RMS_radius(proj))` in
  2-PC space (`trajectory_length.py:88-120`). Rationale: raw arc length is inflated by a larger
  hidden-state norm (a scale artifact), so `L_norm` compares path *geometry* across architectures on
  different latent scales.
- PCA projection makes latent paths of different dimensionality comparable; variance-explained is
  reported so the 2-D view's faithfulness is explicit.
- **All measurements at random initialisation** (Hasani's trajectory-length protocol).

## C2. Experimental setup & two tracks
This is a **separate experiment** from the forecasting runs (Part B) — no forecasting, no walk-forward,
no Supabase run; it measures latent geometry at random init. Runner:
`python -m analysis.run_expressivity_chapter --tickers AAPL SPY JPM --seeds 10 --widths 16 --steps 100`.

- **Input construction** (shared): per ticker, the standardized feature matrix is cut into
  **non-overlapping, tail-aligned windows of `steps = 100` trading days** (`n_rows // 100` ≈ **25
  windows** per ticker; oldest remainder rows dropped). Each window is **standardized per-window**
  (`run_expressivity_chapter.py:63-76`). Feature counts differ by ticker (AAPL 487, JPM 387, SPY 216 —
  SPY drops `^GSPC` + has no SEC fundamentals), so `L_norm` magnitudes are compared **within** ticker.
- **Two tracks:**
  - **Canonical track** — faithful port of Hasani's MATLAB trajectory-length apparatus
    (`hasani_trajectory.py`, vendored from `raminmh/liquid_time_constant_networks` under `vendor/`).
    Three cells sharing width/init — Neural-ODE `dy/dt = Htanh(W_in·x+b)`, CT-RNN
    `dy/dt = −y/τ + Htanh(W_in·x+b)`, LTC (fused input-dependent-τ form). `Htanh` = clip to [−1,1];
    init weights ~ N(0, σ), biases ~ N(0, σ_b), τ = |N(0, σ_b)|; integrated with **adaptive RK45**
    (`scipy.solve_ivp`, rtol 1e-2 / atol 1e-4) sampled on a uniform grid (`hasani_trajectory.py:104-134`).
    Input = the financial window **PCA-reduced to m = 2 PCs** (Hasani's stiffness regime; parity with
    his synthetic unit-circle input). Reports **raw 2-PC arc length** (Hasani parity) **and** `L_norm`.
    Width parameter **k** (hidden units) set to **100** for the headline.
  - **Bench track** — the **production architectures** (LTC, CfC, CTRNN, LSTM as used in the forecasting
    grid) at random init, **units = 16**, fed the **full standardized feature window** (not PCA-reduced);
    latent extracted per step, then `L_norm` on the first 2 PCs (`run_expressivity_chapter.py:118-135`).
    Divergent random inits yield a non-finite latent, recorded as `finite = False` rather than crashing.
- **Sampling / seeds**: measurements repeated over **10 random-init seeds × ~25 windows** per ticker
  (n ≈ 250 per architecture); medians reported. A finite-guard drops non-finite latents.
- **Reproducibility**: deterministic per seed (re-running reproduces identical medians — verified);
  result summaries committed at `results/expressivity/chapter_bench_summary.csv` (+ canonical /
  trajectory summaries); feature-window caches + figures regenerable via the runner (gitignored).

## C3. Headline finding — the LTC dominance is largely a SCALE artifact
- Canonical, AAPL, 25 non-overlapping windows × 10 seeds, k = 100
  (`expressivity_scale_finding.md:24-40`):
  - RAW arc length (replicates Hasani): LTC/CT-RNN ≈ **12.6×**, LTC/Neural-ODE ≈ **5.9×**.
  - Scale-invariant `L_norm`: LTC/CT-RNN ≈ **1.00×**, LTC/Neural-ODE ≈ **~1.0×**.
- Not a finance quirk — same on Hasani's own synthetic circle (k = 100, 20 seeds): raw 13.8× / 8.8×
  collapse to `L_norm` 1.08× / 1.62×.
- So the order-of-magnitude dominance is mostly latent **magnitude**, not per-radius geometric
  complexity. LTC vs CT-RNN is *entirely* scale; LTC vs Neural-ODE keeps a modest scale-free edge
  (≤1.6× on the circle), erased to ~1.0× on noisy financial input.
- `L_norm` still discriminates (not a wash-out): bench track keeps real spread —
  AAPL **CfC 0.32 > LTC 0.27 > CT-RNN 0.22 > LSTM 0.20** — liquid nets retain a modest scale-free edge
  over LSTM. **This ordering replicates on SPY (0.54 > 0.43 > 0.40 > 0.30) and JPM (0.51 > 0.36 > 0.32
  > 0.28)** at 10 seeds — the architecture ranking is stable, though magnitudes differ by input
  geometry (see C6).

## C4. Robustness (m-sensitivity, AAPL canonical, k = 100, 3 seeds × 25 windows) ✅
- Collapse is **not** an artifact of the lossy m = 2 shadow — `L_norm` LTC/CT-RNN stays ~1.0× at
  m = 2, 4, 8 (up to 90% variance) (`expressivity_scale_finding.md:42-54`).
- LTC's scale-free edge over Neural-ODE grows modestly with input richness (1.02× → 1.23× as m: 2→8)
  but never approaches the raw 6–7×.
- Financial input does **not** go numerically stiff up to m = 8 (100% finite) — so m = 2 is a
  Hasani-parity choice here, not a numerical necessity.

## C5. Framing & caveats (decided)
- Report **both** raw (replicates Hasani) and `L_norm` (parity / modest edge); the **divergence is
  the finding**. Do not declare the dominance "fake" — magnitude-as-expressivity is a defensible
  rebuttal.
- Strengthens the negative-result thesis: the expressivity advantage is **partly illusory (scale)**,
  and — per the bridge in C6 — even where `L_norm` weakly tracks forecast rank, it **does not translate
  into beating the classical AR(1)/SARIMAX baseline**.
  - **⚠️ CAUSAL-FRAMING GUARD (do not overclaim in either direction)**: state expressivity as an
    *unreliable predictor of forecast skill*, **not** as the *cause* of the neural loss, and **not** as
    *uncorrelated* with skill (C6 shows a weak positive rank association). The defensible claim is: the
    theoretically-motivated LTC expressivity edge is (i) largely scale, and (ii) insufficient to
    produce a statistically significant forecasting win over the classical baseline.
- Caveats to state: canonical 2-PC input is a lossy ~48%-variance shadow (PC1 ~32% + PC2 ~18%); bench
  track uses the full feature window; all measurements at random init (untrained) — so the bridge in
  C6 compares an init-time geometry measure against trained forecast skill, an intentional
  cross-reference, not a same-object correlation.
- **⚠️ DEFERRED / SCOPE (largely resolved)**: the expressivity↔forecast-skill bridge is now built for
  **all three tickers** (C6), and the bench `L_norm` ordering + canonical LTC-dominance check both
  reproduced on SPY and JPM. Still AAPL-primary: the full canonical **raw-vs-`L_norm` scale-artifact
  ratio table** (C3) and m-sensitivity (C4) — the SPY/JPM canonical runs confirmed LTC-dominance but
  the headline scale-collapse ratios are reported on AAPL + Hasani's synthetic circle.

## C6. Expressivity → forecast-skill bridge (all 3 tickers) ✅ NEW
- **Question**: does an architecture's scale-invariant latent expressivity (`L_norm`, bench track,
  random init, 10 seeds × 25 windows, units = 16) predict how well it actually forecasts (Run #2a
  RMSE, default variant, mean over 5 seeds)? This is the link Codex flagged as the single
  highest-leverage addition — without it the expressivity chapter reads as a separate paper.
- **Method note**: `L_norm` magnitudes are **not comparable across tickers** (AAPL 487 features, JPM
  387, SPY 216 — different input geometry), so read the bridge **within each ticker** (does the
  `L_norm` ordering predict the RMSE ordering?), not across.
- **Bridge table** — bench `L_norm` order vs Run #2a forecast RMSE order, per ticker. Bench arch ↔
  forecast variant: CfC↔`default_ncp`, LTC↔`ncp`, CT-RNN↔`ctrnn`, LSTM↔`lstm`:

  | Ticker | `L_norm` order (rank) | Forecast RMSE order (best→worst) | Spearman ρ |
  |---|---|---|---|
  | **AAPL** | CfC > LTC > CT-RNN > LSTM | CfC > LTC > LSTM > CT-RNN | **0.80** |
  | **SPY**  | CfC > LTC > CT-RNN > LSTM | CfC > LTC > CT-RNN > LSTM | **1.00** |
  | **JPM**  | CfC > LTC > CT-RNN > LSTM | CfC > LTC > LSTM > CT-RNN | **0.80** |

- **Findings (3-ticker, robust):**
  1. **`L_norm` architecture ordering is stable across all three tickers**: CfC > LTC > CT-RNN > LSTM
     everywhere (10-seed medians). Not an AAPL artifact.
  2. **Expressivity is a moderate-to-strong, consistent positive predictor of neural forecast rank**:
     Spearman ρ = 0.80 / 1.00 / 0.80 (AAPL/SPY/JPM). The two most-expressive architectures (CfC, LTC)
     are the two best neural forecasters on every ticker; only the low-expressivity bottom pair
     (CT-RNN, LSTM) swaps. **⇒ Do NOT claim `L_norm` is uncorrelated with skill — it clearly orders
     the neural field.**
  3. **But expressivity does not lift any neural model past the classical baseline**: each family's
     default variant is **10–17% worse** in RMSE than AR(1)/SARIMAX on all three tickers. Even the
     single best neural model overall — `cfc_pure_ncp` — only edges SARIMAX by **0.6–0.9%** (AAPL
     −0.63%, SPY −0.94%, JPM −0.85%) and that margin is **DM-insignificant (Holm p = 1.0)**: it *ties*
     the baseline (see [[finding_run2_cfc_ties_sarimax]]).
- **Bridge claim (defensible, 3-ticker wording)**: *"Scale-normalized latent trajectory length
  reliably orders the neural architectures by forecast skill (CfC ≈ LTC > CT-RNN ≈ LSTM on all three
  assets, Spearman ρ = 0.80–1.00), yet even the most expressive architecture fails to significantly
  beat a simple AR(1)/SARIMAX baseline. Expressivity ranks the neural field but does not carry any
  neural model past the classical baseline on low-SNR daily returns."*
- **⚠️ SCOPE**: n = 4 architecture points per ticker; init-time `L_norm` (untrained) vs trained RMSE —
  a deliberate cross-reference, not a same-object correlation. The bench `L_norm` for all three
  tickers was computed via `analysis.run_expressivity_chapter --tickers AAPL SPY JPM --seeds 10`
  (features cached at `results/expressivity/`).

---

# PART D — Limitations checklist (consolidate into a Limitations section)
1. **FRED macro release-lag look-ahead** (A3) — documented-by-design, argued conservative.
2. **Historical FRED frequency-inference bug** (A9) — nulled 4 monthly macro features in #0/#1; state
   whether Run #2's fix restores them and whether it affects the headline (memory: headline unaffected).
   *Now snapshot-verified: the pinned `grid-2026-06-11` frames (used by runs #0 **and** #1) fail the
   current data-quality gate with 12 errors each (4 all-NaN macro + 8 daily series downsampled to
   ~monthly); the run #2a/2b frames validate at 0 errors — cite the snapshot validation as the
   evidence artifact.* ★
3. **Run 2b confound** (B4) — stationarization changes exogenous representation as a block on top of 2a.
4. **Seed order-dependence** (A5) — RNG seeded once per (ticker, seed).
5. **Fixed seeds / no per-fit reseed** — kept for cross-run comparability.
6. **SPY feature asymmetry** (A3) — `^GSPC` excluded → fewer features than AAPL/JPM.
7. **Day-1 vs day-21 horizon gap** — evaluation is rolling 1-step-ahead, so a genuine multi-day-ahead
   (day-1 vs day-21) horizon comparison is not recoverable without a new eval mode.
8. **Source-repo divergence** (B3) — NCP wiring / 16 units ≠ reference-paper configuration.
9. **ADF/KPSS target-stationarity claim** (B4) — cite your own test output, not bare assertion.
10. **Fixed 50-epoch / no-early-stopping budget** (A5) — likely examiner attack ("neural models not
    given their best chance"); defend as a deliberate uniform-budget fairness control, not tuning.
11. **Expressivity→skill bridge scope** (C6) — now all 3 tickers, but still only n = 4 architecture
    points per ticker and init-time `L_norm` vs trained RMSE (cross-reference). The rank association is
    consistent and positive (ρ = 0.80–1.00); the load-bearing claim is "expressivity orders the neural
    field but no neural model beats AR(1)/SARIMAX," not a precise correlation estimate.

*Items 12–15 added 2026-07-06 after the post-hoc input-data audit
(`src/docs/audit_findings_2026-07-06.md`); all four are exogenous-input defects, symmetric across
models, and cannot favor the exog-free AR(1) headline baseline.* ★

12. **SEC fundamental duration mixing** (A2) — income/cash-flow "quarterly" features mix 3/6/9/12-month
    durations (JPM: annual ↔ 0 square wave); balance-sheet items clean. ★
13. **FRED macro staleness tail** (A2) — all macro columns are frozen forward-fills over the final
    ≈7 months of every run (fixed download end-date 2025-11-07 vs prices tracking the run date). ★
14. **Run 2b missed one series** (B4) — `us_real_manufacturing_&_trade_sales` stayed a raw level
    (key-sanitization mismatch); "every exogenous feature stationarized" must read "all but one". ★
15. **Fundamental coverage window** (A2) — fundamentals begin ~2021 (last ~20 quarters kept), so the
    first half of the 10-y sample carries mean-filled placeholder fundamentals. ★
16. **Exog-drop robustness cell** — one-cell check (AAPL, seed 42; naive + AR(1) + `cfc_pure_ncp`;
    `scripts/gen_robustness_configs.py`) rerunning the headline pair with FRED macro + SEC fundamentals
    REMOVED (32 feature cols vs 492) vs a full-feature control arm on identical folds (n = 85). The
    treatment arm's pinned input is an exact column-subset of the control arm's frame → target
    byte-identical by construction (two independent Yahoo fetches wobble at ~1e-6 in adjusted prices;
    the subset design removes that confound). Result: see thesis_results.md Part D-AUDIT. ★
