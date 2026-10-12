-- Schema for the BachelorThesisExperiment Supabase project.
-- Paste this into the SQL Editor of a fresh Supabase project.
-- Three tables, each with permissive RLS for the anon role so the existing
-- client (which uses the anon/public key) can insert/select.

-- ─── 1. Per-fold experiment metric rows ──────────────────────────────────────
create table if not exists public.experiment_results (
    uuid                  uuid primary key default gen_random_uuid(),
    created_at            timestamptz       default now(),

    model_type            text   not null,
    params                jsonb,
    train_start           text,
    train_end             text,
    test_start            text,
    test_end              text,
    fold_id               bigint,
    horizon               bigint,

    mse                   double precision,
    mae                   double precision,
    rmse                  double precision,
    smape                 double precision,
    mase                  double precision,
    directional_accuracy  double precision,

    runtime_seconds       double precision,
    ticker                text,
    notes                 text,
    diagnostics           jsonb
);

-- ─── 2. Aggregated in-sample (historic) forecast rows ────────────────────────
create table if not exists public.experiment_historic_forecasts (
    uuid                  uuid primary key default gen_random_uuid(),
    created_at            timestamptz       default now(),

    experiment_id         uuid references public.experiment_results(uuid) on delete cascade,
    forecast_run_id       uuid,
    ticker                text,
    model_type            text,
    horizon               bigint,

    mse                   double precision,
    mae                   double precision,
    rmse                  double precision,
    smape                 double precision,
    mase                  double precision,
    directional_accuracy  double precision,
    coverage              double precision,
    mean_pi_width         double precision,
    fold_id               bigint,
    notes                 text
);

-- ─── 3. Per-step future forecast rows (batch insert) ─────────────────────────
create table if not exists public.experiment_future_forecasts (
    uuid                  uuid primary key default gen_random_uuid(),
    created_at            timestamptz       default now(),

    experiment_id         uuid,
    forecast_run_id       uuid,
    ticker                text,
    model_type            text,
    timestamp             text,           -- ISO-formatted string from Python side
    horizon_step          bigint,

    y_forecast            double precision,
    y_lower               double precision,
    y_upper               double precision,
    actual                double precision,
    error                 double precision,
    abs_error             double precision,
    sq_error              double precision,
    direction_correct     boolean,
    in_interval           boolean,
    pi_width              double precision,

    fold_id               bigint,
    horizon               bigint,
    notes                 text
);

-- ─── 4. Per-(ticker, run) aggregated summary blob ────────────────────────────
-- Captures the walk-forward summary that main.py currently only writes to disk
-- as src/results/walk_forward/{ticker}_wf_summary.json. Stored here so cloud
-- runs survive VM teardown. Also bundles any rows from the resilient logger's
-- local JSONL fallback so a paused-Supabase incident can be reconciled by
-- replaying fallback_contents from this row.
create table if not exists public.experiment_summaries (
    uuid                       uuid primary key default gen_random_uuid(),
    created_at                 timestamptz       default now(),

    ticker                     text   not null,
    run_id                     text,           -- shared across all tickers in one main() invocation
    git_commit                 text,           -- best-effort git rev-parse HEAD at run start

    n_folds_total              bigint,
    n_folds_primary_aligned    bigint,
    initial_window             bigint,
    horizon                    bigint,
    seed                       bigint,

    metrics                    jsonb,
    primary_significance_table jsonb,
    secondary_significance_table jsonb,
    diagnostics                jsonb,
    summary_full               jsonb,           -- entire summary dict as written to disk
    fallback_contents          jsonb            -- {table_name: [row, ...]} from logger_fallback/*.jsonl
);

-- ─── Indexes for the queries you'll write later ──────────────────────────────
create index if not exists experiment_results_ticker_model_fold_idx
    on public.experiment_results (ticker, model_type, fold_id);

create index if not exists experiment_future_forecasts_run_idx
    on public.experiment_future_forecasts (forecast_run_id);

-- ─── RLS: enable + permissive policies for the anon (public) key ─────────────
-- The client uses the anon/public key, so anon needs insert + select. If you
-- want stricter isolation later, swap these for service-role-only policies and
-- have the client authenticate as a service role.
alter table public.experiment_results               enable row level security;
alter table public.experiment_historic_forecasts    enable row level security;
alter table public.experiment_future_forecasts      enable row level security;
alter table public.experiment_summaries             enable row level security;

create policy "anon insert experiment_results"
    on public.experiment_results
    for insert to anon
    with check (true);

create policy "anon select experiment_results"
    on public.experiment_results
    for select to anon
    using (true);

create policy "anon insert experiment_historic_forecasts"
    on public.experiment_historic_forecasts
    for insert to anon
    with check (true);

create policy "anon select experiment_historic_forecasts"
    on public.experiment_historic_forecasts
    for select to anon
    using (true);

create policy "anon insert experiment_future_forecasts"
    on public.experiment_future_forecasts
    for insert to anon
    with check (true);

create policy "anon select experiment_future_forecasts"
    on public.experiment_future_forecasts
    for select to anon
    using (true);

create policy "anon insert experiment_summaries"
    on public.experiment_summaries
    for insert to anon
    with check (true);

create policy "anon select experiment_summaries"
    on public.experiment_summaries
    for select to anon
    using (true);
