"""
Optional cross-run Diebold-Mariano merge (run #2, eng-review Section 5 / Codex #3-4).

SECONDARY analysis. The run #2 headline (variant-vs-AR(1)) is computed IN-RUN on
identical folds — this module is only for the extended table that adds run #1's
EXPENSIVE neural baselines (LSTM/CTRNN/NeuralODE) to the run #2 variant family.

Pairing validity is NOT assumed from a shared INPUT_GROUP_ID alone (that proves
identical input frames, not an identical fold protocol). Before any DM test we
assert, per (seed, ticker, fold_id, t), that every run_key reports the SAME
y_true — a hard-fail on mismatch, never a silent wrong p-value.

The pure functions below operate on already-loaded prediction frames (the schema
written by ``main._persist_predictions``). ``_main`` is the thin Supabase I/O
shell that fetches those frames; it is not unit-tested (# pragma: no cover).

Prediction frame columns (run #2):
    run_group_id, run_id, seed, ticker, model, variant, run_key, condition,
    fold_id, t, y_true, y_pred, residual
Run #1 parquets predate the variant columns, so ``run_key`` is synthesized from
``model``+``condition`` when absent (back-compat).
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from evaluation.evaluator import Evaluator


def ensure_run_key(df: pd.DataFrame) -> pd.DataFrame:
    """Guarantee a ``run_key`` column. Run #1's older prediction snapshots stored
    only ``model``+``condition`` (no variant), so synthesize the same key the
    orchestrator would have used: ``f"{model}_{condition}"``."""
    df = df.copy()
    if "run_key" not in df.columns or df["run_key"].isna().any():
        synth = df["model"].astype(str) + "_" + df["condition"].astype(str)
        if "run_key" in df.columns:
            df["run_key"] = df["run_key"].fillna(synth)
        else:
            df["run_key"] = synth
    return df


def assert_y_true_consistent(df: pd.DataFrame) -> None:
    """Raise if any two run_keys disagree on y_true at a shared (seed, ticker,
    fold_id, t). This is the real pairing guard for cross-run DM."""
    keys = ["seed", "ticker", "fold_id", "t"]
    # For each coordinate, the set of y_true values across run_keys must be one
    # value (within fp tolerance). Group and check spread.
    grp = df.groupby(keys)["y_true"]
    spread = grp.agg(lambda s: float(np.nanmax(s) - np.nanmin(s)))
    bad = spread[spread > 1e-9]
    if len(bad) > 0:
        first = bad.index[0]
        raise ValueError(
            f"y_true mismatch across run_keys at (seed,ticker,fold_id,t)={first} "
            f"(spread={bad.iloc[0]:.3e}); cross-run DM pairing is invalid. "
            f"{len(bad)} coordinate(s) disagree — check INPUT_GROUP_ID and fold protocol."
        )


def _common_fold_ids(df: pd.DataFrame, run_keys: List[str]) -> List[int]:
    """Fold ids present for EVERY run_key in the family (intersection)."""
    sets = [set(df.loc[df["run_key"] == rk, "fold_id"].unique()) for rk in run_keys]
    if not sets:
        return []
    common = set.intersection(*sets)
    return sorted(int(f) for f in common)


def build_evaluator(df_one: pd.DataFrame, run_keys: List[str],
                    horizon: int = 1) -> Evaluator:
    """Build an Evaluator for a single (seed, ticker) slice over ``run_keys`` on
    their common folds.

    Within each common fold, every run_key must share the SAME set of ``t``
    coordinates — otherwise the positional ``sort_values("t")`` comparison would
    silently misalign forecasts. We hard-fail on a per-fold coordinate mismatch
    (the "fold ids + lengths" half of the pairing guard; assert_y_true_consistent
    covers the values)."""
    common = _common_fold_ids(df_one, run_keys)
    by_key = {rk: df_one[df_one["run_key"] == rk] for rk in run_keys}
    ev = Evaluator(horizon=horizon)
    for fold_id in common:
        ref_t = None
        for rk in run_keys:
            f = by_key[rk][by_key[rk]["fold_id"] == fold_id].sort_values("t")
            t_arr = f["t"].to_numpy()
            if ref_t is None:
                ref_t = t_arr
            elif not np.array_equal(t_arr, ref_t):
                raise ValueError(
                    f"fold {fold_id}: run_key {rk!r} has t-coordinates "
                    f"{t_arr.tolist()} != reference {ref_t.tolist()} — cross-run "
                    "alignment would be positionally wrong.")
            ev.add_fold(rk, int(fold_id),
                        f["y_true"].to_numpy(dtype=float),
                        f["y_pred"].to_numpy(dtype=float),
                        f["residual"].to_numpy(dtype=float))
    return ev


def merge_dm_per_seed(frames: List[pd.DataFrame],
                      primary_run_keys: Optional[List[str]] = None,
                      horizon: int = 1) -> Dict[tuple, pd.DataFrame]:
    """Merge run #2 + run #1 prediction frames and compute a per-(seed, ticker)
    DM significance table over the union of run_keys.

    primary_run_keys restricts the Holm-Bonferroni family (e.g. the variant +
    AR(1) contrast set) so the corrected family is the predeclared one, not the
    full all-pairs blow-up. None ⇒ all run_keys.

    Returns {(seed, ticker): significance_table_df}.
    """
    df = ensure_run_key(pd.concat(frames, ignore_index=True))
    assert_y_true_consistent(df)

    out: Dict[tuple, pd.DataFrame] = {}
    for (seed, ticker), df_one in df.groupby(["seed", "ticker"]):
        run_keys = sorted(df_one["run_key"].unique())
        fam = [k for k in (primary_run_keys or run_keys) if k in run_keys]
        # Align folds over the FAMILY only: a secondary model with missing folds
        # must not shrink the common-fold set used for the corrected family DM.
        ev = build_evaluator(df_one, fam, horizon=horizon)
        out[(int(seed), str(ticker))] = ev.get_significance_table(primary_models=fam)
    return out


def _main(run2_group: str, run1_group: str, tickers: List[str]) -> None:  # pragma: no cover - thin Supabase I/O
    """Fetch run #2 variant forecasts + run #1 neural-baseline forecasts from
    Supabase Storage and print the merged per-seed DM tables.

    Forecasts live as per-(group, ticker, run_id) parquet snapshots; we resolve
    each run's predictions via the experiment_summaries snapshot_uri and download
    + deserialize them (Codex #4: this repo uses parquet_bytes_to_frame, and
    there is no provenance.read_parquet)."""
    import os as _os
    from supabase import create_client
    from provenance import parquet_bytes_to_frame
    from loggers.supabase_logger import SupabaseLogger

    client = create_client(_os.environ["SUPABASE_URL"], _os.environ["SUPABASE_KEY"])
    logger = SupabaseLogger(_os.environ["SUPABASE_URL"], _os.environ["SUPABASE_KEY"])

    _NEURAL_BASELINES = {"lstm_transformed", "ctrnn_transformed", "neural_ode_transformed"}
    _BUCKET = "experiment-snapshots"

    def _load_group(group: str) -> List[pd.DataFrame]:
        rows = (client.table("experiment_summaries")
                .select("ticker,run_id,snapshot_uri")
                .eq("run_group_id", group).execute().data)
        frames = []
        for r in rows:
            if r.get("ticker") not in tickers:
                continue
            uri = (r.get("snapshot_uri") or {}).get("predictions")
            if not uri:
                continue
            path = uri.split(f"{_BUCKET}/", 1)[-1]
            data = logger.download_snapshot(_BUCKET, path)
            frames.append(parquet_bytes_to_frame(data))
        return frames

    run2_frames = [ensure_run_key(f) for f in _load_group(run2_group)]
    run1_all = _load_group(run1_group)
    # Keep only the expensive neural baselines from run #1.
    run1_frames = [ensure_run_key(f) for f in run1_all]
    run1_frames = [f[f["run_key"].isin(_NEURAL_BASELINES)] for f in run1_frames]

    # Run #2 is now SELF-CONTAINED: config.yaml runs lstm/ctrnn/neural_ode in-run,
    # so the neural baselines already live in run2_frames. Merging run #1's copies
    # on top would duplicate the same (seed, ticker, run_key, t) coordinates and
    # break the alignment guard. Drop any run #1 baseline whose run_key already
    # appears in run #2 — the merge then degrades to a no-op for self-contained
    # groups and still works for the legacy variants-only run #2 layout.
    run2_keys = set()
    for f in run2_frames:
        if "run_key" in f.columns:
            run2_keys.update(f["run_key"].unique())
    if run2_keys:
        run1_frames = [f[~f["run_key"].isin(run2_keys)] for f in run1_frames]

    tables = merge_dm_per_seed(run2_frames + run1_frames)
    for (seed, ticker), tbl in sorted(tables.items()):
        print(f"\n=== Merged DM ({ticker}, seed {seed}) ===")
        print(tbl.to_string(index=False))


if __name__ == "__main__":  # pragma: no cover
    import sys as _sys
    if len(_sys.argv) < 4:
        print("usage: python -m analysis.merge_dm <run2_group> <run1_group> <ticker[,ticker...]>")
        raise SystemExit(2)
    _main(_sys.argv[1], _sys.argv[2], _sys.argv[3].split(","))
