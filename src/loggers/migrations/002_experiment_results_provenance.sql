-- Migration 002: provenance + idempotency for experiment_results.
-- Apply via the Supabase SQL Editor. Idempotent — safe to re-run.
--
-- WHY: per-fold rows previously had no run_id/condition/seed, and were logged
-- from inside each model's predict() using a class-global GLOBAL_FOLD_COUNTER
-- (not the walk-forward fold index). The 1st full grid's crash-resume also
-- produced DUPLICATE AAPL rows. This migration adds the provenance columns and
-- a UNIQUE index so the new centralized logging path (main.py) can upsert
-- idempotently on a crash-resume re-run.
--
-- NULL note: Postgres treats NULLs as DISTINCT in a unique index, so the
-- legacy rows below (run_id/condition/seed all NULL) never collide with each
-- other or with new rows — a FULL (non-partial) unique index is therefore
-- safe AND is usable as a PostgREST on_conflict arbiter (a partial index's
-- predicate is not reliably targetable via on_conflict).

-- 1. Provenance columns (nullable, back-compatible with legacy rows).
alter table public.experiment_results add column if not exists run_id    text;
alter table public.experiment_results add column if not exists condition text;
alter table public.experiment_results add column if not exists seed      bigint;

-- 2. One-off dedupe of the legacy duplicate rows (run_id IS NULL) produced by
--    the 2026-06-09 crash-resume. Keep the earliest-created row per
--    (ticker, model_type, fold_id); delete the rest. Scoped to run_id IS NULL
--    so it can never touch rows written by the new (run_id-stamped) path.
delete from public.experiment_results a
using public.experiment_results b
where a.run_id is null
  and b.run_id is null
  and a.ticker      = b.ticker
  and a.model_type  = b.model_type
  and a.fold_id     = b.fold_id
  and a.created_at  > b.created_at;   -- delete the newer duplicate, keep oldest

-- 3. Full unique index = upsert arbiter for the centralized logging path.
create unique index if not exists experiment_results_provenance_uidx
    on public.experiment_results
    (run_id, ticker, model_type, condition, seed, fold_id);
