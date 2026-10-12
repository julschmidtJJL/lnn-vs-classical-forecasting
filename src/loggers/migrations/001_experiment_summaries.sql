-- Migration 001: experiment_summaries table.
-- Apply via the Supabase SQL Editor. Idempotent — safe to re-run.
--
-- SECURITY NOTE: RLS is permissive for the `anon` role (insert + select),
-- mirroring the existing tables in supabase_schema.sql. This means anyone
-- with the project's public anon key can read every summary, including
-- `summary_full` (all per-model metrics + diagnostics) and
-- `fallback_contents` (every row that landed in the local resilience
-- fallback during a run). Acceptable for a private bachelor-thesis
-- project where the anon key is treated as semi-public. NOT a production
-- posture: for that, swap to service-role-only policies and authenticate
-- the client accordingly.
--
-- Captures the walk-forward summary that main.py also writes to disk as
-- src/results/walk_forward/{ticker}_wf_summary.json. Stored here so cloud
-- VM teardown doesn't lose the headline numbers. Also bundles any rows
-- that landed in the resilient logger's local JSONL fallback so a
-- paused-Supabase incident can be reconciled by replaying fallback_contents.

create table if not exists public.experiment_summaries (
    uuid                       uuid primary key default gen_random_uuid(),
    created_at                 timestamptz       default now(),

    ticker                     text   not null,
    run_id                     text,
    git_commit                 text,

    n_folds_total              bigint,
    n_folds_primary_aligned    bigint,
    initial_window             bigint,
    horizon                    bigint,
    seed                       bigint,

    metrics                    jsonb,
    primary_significance_table jsonb,
    secondary_significance_table jsonb,
    diagnostics                jsonb,
    summary_full               jsonb,
    fallback_contents          jsonb
);

alter table public.experiment_summaries enable row level security;

drop policy if exists "anon insert experiment_summaries" on public.experiment_summaries;
create policy "anon insert experiment_summaries"
    on public.experiment_summaries
    for insert to anon
    with check (true);

drop policy if exists "anon select experiment_summaries" on public.experiment_summaries;
create policy "anon select experiment_summaries"
    on public.experiment_summaries
    for select to anon
    using (true);
