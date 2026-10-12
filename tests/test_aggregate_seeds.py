"""Tests for the multi-seed aggregation core (src/analysis/aggregate_seeds.py)."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.analysis.aggregate_seeds import (
    aggregate_metrics,
    aggregate_significance,
    dedup_latest,
)


def _rows():
    # AAPL, two seeds. Classical (sarimax) identical across seeds; neural varies.
    return [
        {"ticker": "AAPL", "seed": 42,
         "metrics": {"sarimax_transformed": {"mae_mean": 0.011, "rmse_mean": 0.015,
                                             "directional_accuracy_mean": 0.48},
                     "lstm_transformed": {"mae_mean": 0.012, "rmse_mean": 0.016,
                                          "directional_accuracy_mean": 0.51}},
         "primary_significance_table": [
             {"model_a": "naive_transformed", "model_b": "lstm_transformed", "significant": 1},
             {"model_a": "naive_transformed", "model_b": "sarimax_transformed", "significant": 1},
         ]},
        {"ticker": "AAPL", "seed": 43,
         "metrics": {"sarimax_transformed": {"mae_mean": 0.011, "rmse_mean": 0.015,
                                             "directional_accuracy_mean": 0.48},
                     "lstm_transformed": {"mae_mean": 0.014, "rmse_mean": 0.018,
                                          "directional_accuracy_mean": 0.49}},
         "primary_significance_table": [
             {"model_a": "naive_transformed", "model_b": "lstm_transformed", "significant": 0},
             {"model_a": "naive_transformed", "model_b": "sarimax_transformed", "significant": 1},
         ]},
    ]


def test_aggregate_metrics_mean_std_across_seeds():
    agg = aggregate_metrics(_rows())
    # classical is seed-invariant → std ~0
    assert agg[("AAPL", "sarimax_transformed")]["mae_mean"]["std"] == pytest.approx(0.0)
    # neural varies → mean of [0.012, 0.014]
    lstm = agg[("AAPL", "lstm_transformed")]["mae_mean"]
    assert lstm["mean"] == pytest.approx(0.013)
    assert lstm["std"] > 0
    assert lstm["n_seeds"] == 2


def test_aggregate_significance_k_of_n():
    sig = aggregate_significance(_rows())
    # lstm beat naive in 1 of 2 seeds; sarimax in 2 of 2
    assert sig[("AAPL", "naive_transformed", "lstm_transformed")] == {"n_significant": 1, "n_seeds": 2}
    assert sig[("AAPL", "naive_transformed", "sarimax_transformed")] == {"n_significant": 2, "n_seeds": 2}


def test_aggregate_metrics_uses_sample_std():
    # [0.012, 0.014]: SAMPLE std = sqrt(2e-6) ≈ 0.0014142 (n-1 denominator);
    # the old population std (pstdev, n denominator) would be 0.001. Pin the
    # sample value so a regression back to pstdev is caught.
    lstm = aggregate_metrics(_rows())[("AAPL", "lstm_transformed")]["mae_mean"]
    assert lstm["std"] == pytest.approx(0.0014142, abs=1e-6)


def test_aggregate_metrics_single_seed_std_zero():
    # One seed → no variability observable; sample std would raise, so we
    # report 0.0 rather than crashing.
    one = aggregate_metrics(_rows()[:1])[("AAPL", "lstm_transformed")]["mae_mean"]
    assert one["std"] == 0.0
    assert one["n_seeds"] == 1


def test_aggregate_significance_denominator_counts_all_seeds():
    # 3 seeds for AAPL, but the lstm pair only appears (and is significant) in 2
    # of them — the third seed dropped/excluded lstm. n_seeds must be 3 (distinct
    # seeds), so the pair reads 2/3, NOT a misleading 2/2 that hides the failed
    # seed. The sarimax pair is present in all 3.
    rows = [
        {"ticker": "AAPL", "seed": 1, "metrics": {},
         "primary_significance_table": [
             {"model_a": "naive_transformed", "model_b": "lstm_transformed", "significant": 1},
             {"model_a": "naive_transformed", "model_b": "sarimax_transformed", "significant": 1},
         ]},
        {"ticker": "AAPL", "seed": 2, "metrics": {},
         "primary_significance_table": [
             {"model_a": "naive_transformed", "model_b": "lstm_transformed", "significant": 1},
             {"model_a": "naive_transformed", "model_b": "sarimax_transformed", "significant": 0},
         ]},
        {"ticker": "AAPL", "seed": 3, "metrics": {},  # lstm pair missing this seed
         "primary_significance_table": [
             {"model_a": "naive_transformed", "model_b": "sarimax_transformed", "significant": 1},
         ]},
    ]
    sig = aggregate_significance(rows)
    assert sig[("AAPL", "naive_transformed", "lstm_transformed")] == {"n_significant": 2, "n_seeds": 3}
    assert sig[("AAPL", "naive_transformed", "sarimax_transformed")] == {"n_significant": 2, "n_seeds": 3}


def _dup_rows():
    # AAPL seed 42 written twice (a resume re-wrote the seed). The earlier write
    # is partial/wrong (mae 0.099); the later write is correct (mae 0.012).
    # Seed 43 appears once. Without dedup, seed 42 would be counted twice.
    early = {"ticker": "AAPL", "seed": 42, "created_at": "2026-06-11T08:00:00Z",
             "metrics": {"lstm_transformed": {"mae_mean": 0.099}},
             "primary_significance_table": [
                 {"model_a": "naive_transformed", "model_b": "lstm_transformed", "significant": 1}]}
    late = {"ticker": "AAPL", "seed": 42, "created_at": "2026-06-11T20:00:00Z",
            "metrics": {"lstm_transformed": {"mae_mean": 0.012}},
            "primary_significance_table": [
                {"model_a": "naive_transformed", "model_b": "lstm_transformed", "significant": 0}]}
    seed43 = {"ticker": "AAPL", "seed": 43, "created_at": "2026-06-12T06:00:00Z",
              "metrics": {"lstm_transformed": {"mae_mean": 0.014}},
              "primary_significance_table": [
                  {"model_a": "naive_transformed", "model_b": "lstm_transformed", "significant": 0}]}
    return [early, late, seed43]


def test_dedup_latest_keeps_most_recent_per_ticker_seed():
    deduped = dedup_latest(_dup_rows())
    # one row per (ticker, seed): seed 42 and seed 43
    keys = sorted((r["ticker"], r["seed"]) for r in deduped)
    assert keys == [("AAPL", 42), ("AAPL", 43)]
    # the kept seed-42 row is the LATER write (mae 0.012), not the stale 0.099
    s42 = next(r for r in deduped if r["seed"] == 42)
    assert s42["metrics"]["lstm_transformed"]["mae_mean"] == 0.012


def test_dedup_latest_prevents_seed_double_count_in_metrics():
    # Without dedup, aggregate_metrics sees 3 rows → n_seeds=3 and a mean
    # polluted by the stale 0.099. After dedup it's 2 clean seeds.
    lstm = aggregate_metrics(dedup_latest(_dup_rows()))[("AAPL", "lstm_transformed")]["mae_mean"]
    assert lstm["n_seeds"] == 2
    assert lstm["mean"] == pytest.approx(0.013)  # mean of [0.012, 0.014]


def test_dedup_latest_untimestamped_row_loses_to_timestamped():
    # A row missing created_at must not beat a real timestamped write.
    no_ts = {"ticker": "AAPL", "seed": 42,
             "metrics": {"lstm_transformed": {"mae_mean": 0.099}},
             "primary_significance_table": []}
    with_ts = {"ticker": "AAPL", "seed": 42, "created_at": "2026-06-11T20:00:00Z",
               "metrics": {"lstm_transformed": {"mae_mean": 0.012}},
               "primary_significance_table": []}
    deduped = dedup_latest([no_ts, with_ts])
    assert len(deduped) == 1
    assert deduped[0]["metrics"]["lstm_transformed"]["mae_mean"] == 0.012


def test_dedup_latest_noop_when_already_unique():
    # Already one row per (ticker, seed) → unchanged count, aggregation intact.
    assert len(dedup_latest(_rows())) == 2


def test_dedup_latest_compares_across_iso_formats():
    # Real Supabase rows use "+00:00"; a fixture/other source may use "Z".
    # Lexicographically "...Z" > "...+00:00", which would wrongly pick the
    # earlier "Z" row. Parsing to datetimes must pick the genuinely later one.
    z_early = {"ticker": "AAPL", "seed": 42, "run_id": "a",
               "created_at": "2026-06-11T08:00:00Z",
               "metrics": {"lstm_transformed": {"mae_mean": 0.099}},
               "primary_significance_table": []}
    offset_late = {"ticker": "AAPL", "seed": 42, "run_id": "b",
                   "created_at": "2026-06-11T20:00:00+00:00",
                   "metrics": {"lstm_transformed": {"mae_mean": 0.012}},
                   "primary_significance_table": []}
    deduped = dedup_latest([z_early, offset_late])
    assert len(deduped) == 1
    assert deduped[0]["metrics"]["lstm_transformed"]["mae_mean"] == 0.012


def test_dedup_latest_tie_broken_by_run_id_deterministically():
    # Identical created_at → result must not depend on input order. run_id
    # breaks the tie (higher run_id wins), both orderings agree.
    a = {"ticker": "AAPL", "seed": 42, "run_id": "run-a",
         "created_at": "2026-06-11T20:00:00+00:00",
         "metrics": {"lstm_transformed": {"mae_mean": 0.011}},
         "primary_significance_table": []}
    b = {"ticker": "AAPL", "seed": 42, "run_id": "run-b",
         "created_at": "2026-06-11T20:00:00+00:00",
         "metrics": {"lstm_transformed": {"mae_mean": 0.022}},
         "primary_significance_table": []}
    forward = dedup_latest([a, b])
    backward = dedup_latest([b, a])
    assert len(forward) == len(backward) == 1
    assert forward[0]["run_id"] == backward[0]["run_id"] == "run-b"
