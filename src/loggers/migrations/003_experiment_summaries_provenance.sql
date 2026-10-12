-- Migration 003: provenance columns + summary idempotency for experiment_summaries.
-- Apply via the Supabase SQL Editor. Idempotent — safe to re-run.
--
-- WHY:
--  - config_snapshot / env_manifest: capture the fully-resolved config and the
--    runtime environment (python + key package versions) so a run is
--    reconstructable, not just its metrics.
--  - data_fingerprint / snapshot_uri: SHA-256 canonical fingerprint of each
--    (ticker, condition) input frame + the Storage path of the parquet snapshot
--    (see Workstream A). Lets us PROVE which inputs produced the numbers.
--  - run_group_id: groups the per-seed sub-runs of one multi-seed experiment
--    (run_id = {run_group_id}#s{seed}); the input snapshot is stored once per
--    group (seed-independent).
--  - unique (run_id, ticker): makes log_summary upsert idempotent on resume.

alter table public.experiment_summaries add column if not exists run_group_id    text;
alter table public.experiment_summaries add column if not exists config_snapshot jsonb;
alter table public.experiment_summaries add column if not exists env_manifest    jsonb;
alter table public.experiment_summaries add column if not exists data_fingerprint jsonb;
alter table public.experiment_summaries add column if not exists snapshot_uri    jsonb;

-- Dedupe any pre-existing (run_id, ticker) duplicates before the unique index
-- (keep earliest). Scoped to non-null run_id so legacy NULL rows are untouched.
delete from public.experiment_summaries a
using public.experiment_summaries b
where a.run_id is not null
  and a.run_id = b.run_id
  and a.ticker = b.ticker
  and a.created_at > b.created_at;

create unique index if not exists experiment_summaries_run_ticker_uidx
    on public.experiment_summaries (run_id, ticker);
