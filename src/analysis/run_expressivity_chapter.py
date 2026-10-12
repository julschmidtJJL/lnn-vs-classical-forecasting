"""Chapter-grade expressivity aggregation: 3 tickers x non-overlapping windows.

Builds the run-#1/#2-INDEPENDENT half of the thesis "open the black box" chapter
on already-available data (no trained checkpoints needed -- expressivity is an
init-time architectural property, the Hasani framing). Two tracks, two input
protocols, reported separately:

  CANONICAL track  -- the faithful Hasani apparatus (Neural-ODE, CT-RNN, LTC
    cells sharing weights/width) fed a financial input reduced to m=2 PCs (the
    documented stiffness regime). Answers: does LTC expressivity DOMINANCE
    replicate on financial data? Reports raw 2-PC arc length (Hasani parity) AND
    the scale-invariant L_norm.

  BENCH track  -- the production architectures (LTC, CfC, CTRNN, LSTM via
    ncps / the grid cells) at deterministic init, fed the FULL standardized
    feature window they would actually consume. Reports L_norm per arch. This is
    the expressivity x-axis the forecast-skill bridge joins to once run #1/#2
    land (the skill y-axis is deliberately NOT computed here).

Comparability fix (eng-review): raw arc length is not comparable across model
classes (a larger hidden-state norm inflates the path -- a scale artifact), so
the cross-architecture metric is L_norm = arc_length / ((T-1)*RMS_radius) in
2-PC space (scale-invariant; see trajectory_length.trajectory_length_normalized).

Windows are NON-OVERLAPPING, tail-aligned, length ``--steps`` -- roughly
independent estimates over the ~10y series (honest CIs), mirroring the grid's
walk-forward spirit and removing the single-window cherry-pick critique.

Run from ``src/``:
    python -m analysis.run_expressivity_chapter                 # AAPL SPY JPM
    python -m analysis.run_expressivity_chapter --tickers AAPL  # one ticker
    python -m analysis.run_expressivity_chapter --seeds 3 --no-bench   # quick
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from analysis.hasani_trajectory import arc_length_2pc, integrate_models
from analysis.expressivity_extract import build_bench_model, extract_latent
from analysis.trajectory_length import trajectory_length, trajectory_length_normalized
from analysis.run_financial_trajectory import (
    _OUT_DIR,
    _load_config,
    materialize_features,
)

_CANON_MODELS = ["node", "ctrnn", "ltc"]
_BENCH_ARCHS = ["ltc", "cfc", "ctrnn", "lstm"]
_MODEL_LABELS = {
    "node": "Neural-ODE", "ctrnn": "CT-RNN", "ltc": "LTC", "cfc": "CfC", "lstm": "LSTM",
}


# ─── windows ──────────────────────────────────────────────────────────────────

def nonoverlap_slices(n_rows: int, steps: int) -> List[slice]:
    """Tail-aligned non-overlapping windows of length ``steps``.

    Returns ``n_rows // steps`` slices; the oldest remainder rows are dropped so
    the most recent window ends at the series tail. Empty if the series is
    shorter than one window.
    """
    n_win = n_rows // steps
    offset = n_rows - n_win * steps
    return [slice(offset + i * steps, offset + (i + 1) * steps) for i in range(n_win)]


def standardize_window(df: pd.DataFrame, sl: slice) -> np.ndarray:
    """Per-window standardized feature matrix, shape (steps, m_full).

    Drops the target (``log_return``) and any non-numeric column, then any column
    not fully observed in this window (tail indicators carry NaN), then z-scores
    each remaining column over the window. This is the shared substrate for both
    tracks: the canonical track PCA-reduces it to m=2; the bench track feeds it
    whole.
    """
    feats = df.drop(columns=["log_return"], errors="ignore").select_dtypes(include=[np.number])
    win = feats.iloc[sl].dropna(axis=1)
    mat = win.to_numpy(dtype=float)
    mu = mat.mean(axis=0)
    sd = mat.std(axis=0)
    sd[sd == 0] = 1.0
    return (mat - mu) / sd


# ─── per-window metrics ───────────────────────────────────────────────────────

def _safe_norm(path: np.ndarray) -> float:
    """L_norm of a latent path, NaN if the path is non-finite (divergent init)."""
    if not np.all(np.isfinite(path)):
        return float("nan")
    L, _ = trajectory_length_normalized(path, k=2)
    return L


def canonical_metrics(canon_in: np.ndarray, seed: int, k: int) -> Dict[str, Dict[str, float]]:
    """Hasani apparatus on a (m, steps) financial input. Returns per-model
    ``{raw, l_norm, finite}`` (raw = 2-PC arc length, Hasani parity)."""
    paths = integrate_models(canon_in, seed=seed, k=k)
    out: Dict[str, Dict[str, float]] = {}
    for model, path in paths.items():
        finite = bool(np.all(np.isfinite(path)))
        out[model] = {
            "raw": arc_length_2pc(path),       # already NaN-guarded
            "l_norm": _safe_norm(path),
            "finite": finite,
        }
    return out


def bench_metrics(bench_X: np.ndarray, seed: int, units: int) -> Dict[str, Dict[str, float]]:
    """Production archs at init on the FULL feature window (steps, n_features).
    Returns per-arch ``{raw, l_norm, finite}``; a divergent random init yields a
    non-finite latent which we record as finite=False rather than crashing."""
    n_features = bench_X.shape[1]
    out: Dict[str, Dict[str, float]] = {}
    for arch in _BENCH_ARCHS:
        model = build_bench_model(arch, units=units, n_features=n_features, seed=seed)
        try:
            latent = extract_latent(model, bench_X, arch)
        except FloatingPointError:
            out[arch] = {"raw": float("nan"), "l_norm": float("nan"), "finite": False}
            continue
        raw, _ = trajectory_length(latent, k=2)
        out[arch] = {"raw": raw, "l_norm": _safe_norm(latent), "finite": True}
    return out


# ─── driver ───────────────────────────────────────────────────────────────────

def run_chapter(tickers: List[str], seeds: int, widths: List[int], steps: int,
                units: int, *, fetch: bool, do_bench: bool, force: bool = False) -> Dict[str, pd.DataFrame]:
    """Aggregate canonical + bench expressivity over tickers x windows x seeds.

    Writes per-row and summary CSVs to results/expressivity/ and returns the two
    summary frames. ``fetch`` controls live materialization of feature frames
    (cached per ticker); set False to use only what's already cached.
    """
    cfg = _load_config()
    canon_rows: List[dict] = []
    bench_rows: List[dict] = []

    for ticker in tickers:
        df = materialize_features(cfg, ticker, force=force) if fetch else \
            pd.read_parquet(_OUT_DIR / f"{ticker.lower()}_features_transformed.parquet")
        slices = nonoverlap_slices(len(df), steps)
        print(f"[chapter] {ticker}: {len(df)} rows -> {len(slices)} non-overlapping windows of {steps}")

        for w_idx, sl in enumerate(slices):
            std_mat = standardize_window(df, sl)                       # (steps, m_full)
            canon_in = PCA(n_components=2).fit_transform(std_mat).T    # (2, steps)
            for seed in range(1, seeds + 1):
                for k in widths:
                    for model, m in canonical_metrics(canon_in, seed, k).items():
                        canon_rows.append({"ticker": ticker, "window": w_idx, "seed": seed,
                                           "k": k, "model": model, **m})
                if do_bench:
                    for arch, m in bench_metrics(std_mat, seed, units).items():
                        bench_rows.append({"ticker": ticker, "window": w_idx, "seed": seed,
                                           "units": units, "model": arch, **m})

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    canon = pd.DataFrame(canon_rows)
    canon.to_csv(_OUT_DIR / "chapter_canonical_per_window.csv", index=False)
    canon_sum = _summarize(canon, group=["ticker", "k", "model"])
    canon_sum.to_csv(_OUT_DIR / "chapter_canonical_summary.csv", index=False)

    out = {"canonical": canon_sum}
    if do_bench and bench_rows:
        bench = pd.DataFrame(bench_rows)
        bench.to_csv(_OUT_DIR / "chapter_bench_per_window.csv", index=False)
        bench_sum = _summarize(bench, group=["ticker", "units", "model"])
        bench_sum.to_csv(_OUT_DIR / "chapter_bench_summary.csv", index=False)
        out["bench"] = bench_sum
    return out


def _summarize(df: pd.DataFrame, group: List[str]) -> pd.DataFrame:
    """Median/mean/std/n of raw + L_norm over the finite rows of each group."""
    ok = df[df["finite"]]
    agg = (ok.groupby(group)
           .agg(raw_median=("raw", "median"), raw_mean=("raw", "mean"),
                lnorm_median=("l_norm", "median"), lnorm_mean=("l_norm", "mean"),
                lnorm_std=("l_norm", "std"), n=("l_norm", "size"))
           .reset_index())
    agg["model_label"] = agg["model"].map(_MODEL_LABELS)
    return agg


def _print_canonical_ordering(canon_sum: pd.DataFrame, widths: List[int]) -> None:
    """Per-ticker LTC-dominance check on the canonical track (L_norm)."""
    print("\n=== Canonical LTC-dominance (L_norm median, finite windows x seeds) ===")
    for ticker in canon_sum["ticker"].unique():
        for k in widths:
            sub = canon_sum[(canon_sum["ticker"] == ticker) & (canon_sum["k"] == k)]
            if sub.empty:
                continue
            vals = {r["model"]: r["lnorm_median"] for _, r in sub.iterrows()}
            ltc, ct, nd = vals.get("ltc", np.nan), vals.get("ctrnn", np.nan), vals.get("node", np.nan)
            dom = (ltc > ct) and (ltc > nd)
            print(f"  {ticker:5s} k={k:<3d}  LTC={ltc:8.3f}  CT-RNN={ct:8.3f}  Neural-ODE={nd:8.3f}"
                  f"   LTC>>both: {'YES' if dom else 'NO'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tickers", nargs="+", default=["AAPL", "SPY", "JPM"])
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--widths", type=int, nargs="+", default=[100, 16])
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--units", type=int, default=16, help="bench-track NCP/hidden width")
    ap.add_argument("--no-fetch", action="store_true", help="use cached parquet only (no live fetch)")
    ap.add_argument("--no-bench", action="store_true", help="canonical track only")
    ap.add_argument("--force-refetch", action="store_true")
    args = ap.parse_args()

    out = run_chapter(args.tickers, args.seeds, args.widths, args.steps, args.units,
                      fetch=not args.no_fetch, do_bench=not args.no_bench, force=args.force_refetch)
    _print_canonical_ordering(out["canonical"], args.widths)
    if "bench" in out:
        print("\n=== Bench L_norm (median, by arch) ===")
        b = out["bench"]
        print(b[["ticker", "units", "model_label", "lnorm_median", "lnorm_std", "n"]].to_string(index=False))
    print(f"\n[chapter] CSVs written to {_OUT_DIR}")


if __name__ == "__main__":
    main()
