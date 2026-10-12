"""
Offline aggregation of a multi-seed run-group (Workstream C / eng-review DM
decision: per-seed DM, reported as a distribution).

Two pure functions over a list of `experiment_summaries` rows (one per
(ticker, seed)):
  - aggregate_metrics: point metrics as mean ± std ACROSS seeds, per
    (ticker, model_condition). Classical models are seed-invariant, so their
    std-across-seeds is ~0 (a useful sanity check).
  - aggregate_significance: for each (ticker, model pair), how many of the N
    seeds found the pair Diebold-Mariano significant ("k of N").

Run:  python -m analysis.aggregate_seeds <run_group_id>
(fetches the group's summary rows from Supabase and prints both tables).
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from statistics import mean, stdev
from typing import Dict, List, Tuple

_MIN_TS = datetime.min.replace(tzinfo=timezone.utc)


def _recency_key(row: Dict) -> Tuple[datetime, str]:
    """Deterministic recency key for a summary row: (created_at, run_id).

    ``created_at`` is parsed to a timezone-aware datetime so the comparison is
    correct across mixed ISO-8601 spellings (``Z`` vs ``+00:00``) and offsets,
    not a fragile lexicographic string compare. A missing or unparseable
    timestamp sorts to the epoch floor, so any real write beats it. ``run_id``
    is the tiebreaker so two rows with identical timestamps resolve the same
    way regardless of the order Supabase returned them in.
    """
    raw = row.get("created_at") or ""
    try:
        ts = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
    except ValueError:
        ts = _MIN_TS
    return ts, str(row.get("run_id") or "")


def dedup_latest(rows: List[Dict]) -> List[Dict]:
    """Collapse to one summary row per (ticker, seed): the most recent by
    ``created_at`` (ties broken by ``run_id``).

    A resume/retry can write more than one summary row for the same
    (ticker, seed) inside a single run_group (see test_resume_summary_exists).
    Left unguarded, ``aggregate_metrics`` would treat each row as a distinct
    seed: ``n_seeds`` inflates and the mean ± std skews toward whichever seed
    was duplicated. We keep the latest write (a resume overwrites an earlier,
    possibly partial, attempt).
    """
    latest: Dict = {}
    for r in rows:
        key = (r.get("ticker"), r.get("seed"))
        prev = latest.get(key)
        if prev is None or _recency_key(r) > _recency_key(prev):
            latest[key] = r
    return list(latest.values())


def aggregate_metrics(rows: List[Dict]) -> Dict:
    """{(ticker, model_condition): {metric: {mean, std, n_seeds}}} across seeds.

    ``std`` is the SAMPLE std across seeds (statistics.stdev), the conventional
    estimate of seed-to-seed variability for a "mean ± std" report. It needs
    n>=2 seeds; with a single seed we report 0.0 (no variability observable),
    which also keeps the classical (seed-invariant) models at std 0.0.
    """
    buckets: Dict = defaultdict(lambda: defaultdict(list))
    for r in rows:
        ticker = r.get("ticker")
        for mc, m in (r.get("metrics") or {}).items():
            for key in ("mae_mean", "rmse_mean", "directional_accuracy_mean"):
                v = m.get(key)
                if v is not None:
                    buckets[(ticker, mc)][key].append(float(v))
    out: Dict = {}
    for (ticker, mc), metricvals in buckets.items():
        out[(ticker, mc)] = {
            key: {"mean": mean(vals),
                  "std": stdev(vals) if len(vals) > 1 else 0.0,
                  "n_seeds": len(vals)}
            for key, vals in metricvals.items()
        }
    return out


def aggregate_significance(rows: List[Dict]) -> Dict:
    """{(ticker, model_a, model_b): {n_significant, n_seeds}} over the per-seed
    primary DM tables ("k of N seeds significant").

    ``n_seeds`` is the number of DISTINCT seeds observed for the ticker (one
    summary row = one seed), NOT the number of seeds in which the pair appeared.
    So if a seed fails or excludes a model, the pair reads e.g. ``4/5`` (one seed
    missing) instead of a misleading ``4/4`` that hides the failed seed.
    """
    seeds_per_ticker: Dict = defaultdict(set)
    pairs_per_ticker: Dict = defaultdict(set)
    n_significant: Dict = defaultdict(int)
    for r in rows:
        ticker = r.get("ticker")
        seeds_per_ticker[ticker].add(r.get("seed"))
        for pair in (r.get("primary_significance_table") or []):
            ab = (pair.get("model_a"), pair.get("model_b"))
            pairs_per_ticker[ticker].add(ab)
            if pair.get("significant") in (1, True):
                n_significant[(ticker, *ab)] += 1
    out: Dict = {}
    for ticker, pairs in pairs_per_ticker.items():
        n_seeds = len(seeds_per_ticker[ticker])
        for ab in pairs:
            key = (ticker, *ab)
            out[key] = {"n_significant": n_significant[key], "n_seeds": n_seeds}
    return out


def _main(run_group_id: str) -> None:  # pragma: no cover - thin Supabase I/O
    import os
    from supabase import create_client

    client = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])
    rows = (client.table("experiment_summaries")
            .select("ticker,seed,run_id,created_at,metrics,primary_significance_table")
            .eq("run_group_id", run_group_id).order("created_at").execute().data)
    if not rows:
        print(f"No summary rows for run_group_id={run_group_id}")
        return

    # Guard against resume/retry writing >1 row per (ticker, seed): keep the
    # latest write before aggregating, so n_seeds and mean ± std are correct.
    n_raw = len(rows)
    rows = dedup_latest(rows)
    if len(rows) < n_raw:
        print(f"(deduped {n_raw - len(rows)} stale summary row(s); "
              f"kept {len(rows)} unique (ticker, seed) rows)")

    print(f"\n=== Point metrics (mean ± std across {len({r['seed'] for r in rows})} seeds) ===")
    for (ticker, mc), mets in sorted(aggregate_metrics(rows).items()):
        mae = mets.get("mae_mean", {})
        da = mets.get("directional_accuracy_mean", {})
        print(f"  {ticker:5} {mc:24} MAE={mae.get('mean', float('nan')):.5f}"
              f"±{mae.get('std', 0):.5f}  DA={da.get('mean', float('nan')):.3f}")

    print("\n=== DM significance distribution (k of N seeds significant) ===")
    for (ticker, a, b), c in sorted(aggregate_significance(rows).items()):
        print(f"  {ticker:5} {a} vs {b}: {c['n_significant']}/{c['n_seeds']}")


if __name__ == "__main__":  # pragma: no cover
    import sys
    _main(sys.argv[1] if len(sys.argv) > 1 else "")
