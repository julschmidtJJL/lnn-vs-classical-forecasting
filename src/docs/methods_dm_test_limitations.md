# DM test limitations + seed-replication framing (methods)

Two statistical caveats surfaced during the run #2 engineering review (Codex
outside-voice, 2026-06-17). Both are methods-section points; neither changes the
run #2 code path.

## 1. Diebold-Mariano at h=1 has no HAC variance correction

`evaluation.evaluator.dm_test_stat` is called with `h=1`. The DM variance is
therefore the plain sample variance of the daily loss differential, with no
Newey-West / HAC adjustment for serial correlation.

Why this matters for this data:
- Daily squared-error differentials inherit **volatility clustering** from
  financial returns, so they are not iid.
- The **walk-forward training windows overlap** (each fold extends the previous
  train set), which can induce additional serial dependence in successive folds'
  errors.

Consequence: the reported DM p-values are likely **optimistic** (too quick to
call a difference significant). The headline direction (AR(1) vs the neural
models) is robust in magnitude, but the exact significance thresholds should be
read with this caveat.

Why it was NOT "fixed" in run #2: a HAC change alters `dm_test_stat` for every
run. Applying it to run #2 alone would make #2's p-values **non-comparable** to
the already-completed runs #0 and #1. The correct fix is a single, deliberate
pass that re-derives the DM tables for #0/#1/#2 from the stored forecast parquet
snapshots with a HAC-corrected estimator. Tracked in `TODOS.md`.

## 2. "k of N seeds significant" is initialization sensitivity, not replication

The multi-seed report counts, per model pair, how many of the N seeds found the
pair DM-significant ("k of N"). This is **not independent statistical
replication**: across seeds the input data and the classical (SARIMAX / naive /
AR(1)) forecasts are **identical** — only the neural-network weight
initialization changes.

So "k of N" measures how stable a neural model's relative performance is to its
initialization seed, i.e. **initialization sensitivity**. It should be reported
and interpreted as such, not as N independent experiments. The classical models
are seed-invariant (their std across seeds is ~0, a useful sanity check, see
`analysis.aggregate_seeds`).
