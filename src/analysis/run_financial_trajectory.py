"""Financial-input trajectory-length experiment (thesis novelty).

Feeds a REAL AAPL feature trajectory into the validated Hasani apparatus
(``hasani_trajectory.trajectory_lengths``) in place of the synthetic unit
circle, and compares the latent arc length traced by the three continuous-time
models (Neural-ODE, CT-RNN, LTC).

Design (see methods notes / memory idea_expressivity_trajectory_length):
  * Input = the most recent ``--steps`` rows of the TRANSFORMED feature frame
    (the same stationary features the grid models consume), each column
    standardized to zero mean / unit variance. Shape (m_features, steps).
    ``--steps`` defaults to 100 for parity with Hasani's 100-step circle.
  * Weights/init are Hasani's exact regime (sigma=2 unscaled, biases N(0,1),
    tau=|N(0,1)|, hard-tanh). The synthetic circle is run alongside on the same
    seeds/width as the validated reference baseline.
  * Internally-valid comparison = CROSS-ARCHITECTURE on the same financial input
    (all three share weights and f_in). Circle-vs-financial differs in input
    dimensionality m, so it is a reference, not a controlled contrast — the
    input-drive scales with m under Hasani's fixed sigma (methods caveat).
  * Widths: k=100 (Hasani parity, where LTC dominance was validated) plus k=16
    (the thesis NCP width) to connect to the actual grid models.

The AAPL feature frame is fetched live (yfinance/FRED/SEC, needs .env) and
cached to parquet so reruns are instant. Run from ``src/``:

    python -m analysis.run_financial_trajectory                # full run
    python -m analysis.run_financial_trajectory --seeds 5      # quick check
    python -m analysis.run_financial_trajectory --no-fetch     # cache only
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import yaml

from analysis.hasani_trajectory import (
    circular_input,
    integrate_models,
    arc_length_2pc,
    arc_length_kd,
)

# Repo layout: this file is src/analysis/, config + outputs live under src/.
_SRC = Path(__file__).resolve().parent.parent
_CONFIG = _SRC / "config.yaml"
_OUT_DIR = _SRC / "results" / "expressivity"
_FEATURE_CACHE = _OUT_DIR / "aapl_features_transformed.parquet"

_TICKER = "AAPL"
_CONDITION = "transformed"
_MODEL_LABELS = {"node": "Neural-ODE", "ctrnn": "CT-RNN", "ltc": "LTC"}


def _load_config() -> dict:
    with open(_CONFIG, "r") as fh:
        return yaml.safe_load(fh)


def _feature_cache(ticker: str) -> Path:
    """Per-ticker transformed-feature parquet cache path."""
    return _OUT_DIR / f"{ticker.lower()}_features_transformed.parquet"


def materialize_features(cfg: dict, ticker: str = _TICKER, *, force: bool = False) -> pd.DataFrame:
    """Build (or load from cache) the TRANSFORMED feature frame for ``ticker``.

    Live fetch reuses main.py's pipeline functions. Cached to a per-ticker
    parquet; pass ``force=True`` to refetch. The hardcoded-Desktop xlsx dump in
    main.py's ``get_features`` is bypassed (we call ``build_df_final`` directly).
    ``ticker`` defaults to AAPL so existing callers are unaffected.
    """
    cache = _feature_cache(ticker)
    if cache.exists() and not force:
        print(f"[fin-traj] loading cached features {cache}")
        return pd.read_parquet(cache)

    print(f"[fin-traj] fetching live {ticker} data (this is the slow step) ...")
    from main import get_internal_data, get_external_data, get_clean_data, build_df_final

    scenario = cfg["experiment"]["scenarios"][0]
    period, interval = scenario["period"], scenario["interval"]
    raw_internal = get_internal_data(ticker=ticker, period=period, interval=interval, cfg=cfg)
    raw_external = get_external_data(period=period, interval=interval, cfg=cfg)
    clean_internal, clean_external = get_clean_data(raw_internal, raw_external, cfg)
    df = build_df_final(clean_internal, clean_external, _CONDITION, cfg, ticker=ticker)

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache)
    print(f"[fin-traj] cached features {df.shape} -> {cache}")
    return df


def select_window(df: pd.DataFrame, steps: int, n_components: int = 2) -> np.ndarray:
    """Most-recent ``steps`` rows of the feature matrix, reduced to a 2-D input
    trajectory dimensionality-matched to Hasani's unit circle.

    Each of the (many) feature columns is first standardized to zero mean / unit
    variance over the window, then the window is projected onto its top
    ``n_components`` principal components (default 2). Feeding 491 raw features
    with Hasani's fixed sigma=2 init saturates hard-tanh and makes the LTC ODE
    stiff; projecting to m=2 keeps the validated regime intact and makes the
    synthetic circle (idealized/periodic) vs financial (real/noisy/non-periodic)
    a controlled contrast at identical m, k and init. Returns (n_components, steps).
    """
    from sklearn.decomposition import PCA

    feats = df.drop(columns=["log_return"]).select_dtypes(include=[np.number])
    win_df = feats.iloc[-steps:]
    # Some engineered features carry NaN (e.g. long-window indicators) at the
    # series tail; PCA cannot ingest them. Drop any column not fully observed
    # over this window before standardizing.
    win_df = win_df.dropna(axis=1)
    print(f"[fin-traj] window features after NaN-drop: {win_df.shape[1]}")
    window = win_df.to_numpy(dtype=float)                         # (steps, m)
    mu = window.mean(axis=0)
    sd = window.std(axis=0)
    sd[sd == 0] = 1.0
    standardized = (window - mu) / sd
    proj = PCA(n_components=n_components).fit_transform(standardized)  # (steps, n_components)
    return proj.T                                                 # (n_components, steps)


def run(seeds: int, widths: List[int], steps: int, *, fetch: bool, force: bool) -> pd.DataFrame:
    cfg = _load_config()
    fin_window = None
    if fetch:
        df = materialize_features(cfg, force=force)
        fin_window = select_window(df, steps)
        print(f"[fin-traj] financial window: m={fin_window.shape[0]} features x {fin_window.shape[1]} steps")
    circle = circular_input(steps)

    rows = []
    inputs = {"circle": circle}
    if fin_window is not None:
        inputs["financial"] = fin_window

    for k in widths:
        for seed in range(1, seeds + 1):
            for input_name, arr in inputs.items():
                # Integrate once, score both the 2-PC shadow (cross-arch metric)
                # and the full k-D path (catches motion in minor components).
                paths = integrate_models(arr, seed=seed, k=k)
                for model, path in paths.items():
                    L = arc_length_2pc(path)
                    L_kd = arc_length_kd(path)
                    rows.append({
                        "input": input_name, "k": k, "seed": seed,
                        "model": model, "arc_length": L, "arc_length_kd": L_kd,
                        "finite": bool(np.isfinite(L)),
                    })
        print(f"[fin-traj] k={k}: {seeds} seeds x {len(inputs)} inputs done")

    per_seed = pd.DataFrame(rows)
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    per_seed.to_csv(_OUT_DIR / "trajectory_lengths_per_seed.csv", index=False)

    # finite-only summary (random-init N-ODE can diverge; guard the aggregate)
    ok = per_seed[per_seed["finite"]]
    summary = (ok.groupby(["input", "k", "model"])
               .agg(mean=("arc_length", "mean"), std=("arc_length", "std"),
                    count=("arc_length", "count"),
                    mean_kd=("arc_length_kd", "mean"), std_kd=("arc_length_kd", "std"))
               .reset_index())
    summary["model_label"] = summary["model"].map(_MODEL_LABELS)
    summary.to_csv(_OUT_DIR / "trajectory_lengths_summary.csv", index=False)

    n_bad = int((~per_seed["finite"]).sum())
    if n_bad:
        print(f"[fin-traj] WARNING: {n_bad} non-finite arc lengths excluded from summary")
    return summary


# Stable model order + a colourblind-safe colour per architecture, shared across
# every figure so the reader keys colour->model once.
_MODEL_ORDER = ["node", "ctrnn", "ltc"]
_MODEL_COLORS = {"node": "#4C72B0", "ctrnn": "#DD8452", "ltc": "#55A868"}


def _apply_style() -> None:
    import seaborn as sns
    sns.set_theme(context="paper", style="ticks", font_scale=1.15)
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "axes.titleweight": "semibold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.family": "sans-serif",
        "pdf.fonttype": 42,  # editable text in vector output (thesis-friendly)
    })


def _save(fig, stem: str) -> None:
    """Write both a vector PDF (for the thesis) and a high-DPI PNG (for preview)."""
    for ext, dpi in (("pdf", None), ("png", 200)):
        fig.savefig(_OUT_DIR / f"{stem}.{ext}", dpi=dpi)


def make_overlay_figure(widths: List[int], steps: int, fetch: bool, *, seed: int = 1) -> None:
    """All three architectures' latent paths in ONE shared frame per (input, width).

    The per-model panels in ``fig_trajectories`` autoscale each axes
    independently, so LTC's huge arc length and Neural-ODE's small one render at
    the same visual size — the very difference the experiment is about gets
    hidden. Here every model is drawn in a single axes on a COMMON scale (2-PC
    space preserves true distances, since PCA only rotates/centres), so a longer
    arc literally sweeps a larger region. Each curve is centred at its own
    origin so only size/shape — not arbitrary PCA offset — is compared.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    _apply_style()
    cfg = _load_config()
    inputs = {"circle": circular_input(steps)}
    if fetch:
        inputs["financial"] = select_window(materialize_features(cfg), steps)
    input_titles = {"circle": "Synthetic circle", "financial": "AAPL trajectory (2-PC)"}

    ncols = len(inputs)
    fig, axes = plt.subplots(len(widths), ncols,
                             figsize=(5.2 * ncols, 4.6 * len(widths)), squeeze=False)
    for r, k in enumerate(widths):
        for c, (iname, arr) in enumerate(inputs.items()):
            ax = axes[r][c]
            paths = integrate_models(arr, seed=seed, k=k)
            for model in _MODEL_ORDER:
                proj = PCA(n_components=2).fit_transform(paths[model])
                proj = proj - proj.mean(axis=0)          # centre; PCA scale is real
                L = arc_length_2pc(paths[model])
                ax.plot(proj[:, 0], proj[:, 1], color=_MODEL_COLORS[model],
                        linewidth=1.8, alpha=0.9,
                        label=f"{_MODEL_LABELS[model]}  ·  L = {L:.0f}")
                ax.scatter(*proj[0], color=_MODEL_COLORS[model], s=22, zorder=3)
            ax.set_aspect("equal", "datalim")
            ax.set_xticks([]); ax.set_yticks([])
            ax.legend(frameon=False, fontsize=9, loc="upper right")
            title = input_titles.get(iname, iname)
            ax.set_title(f"{title}   (k={k})")
    fig.suptitle("Latent trajectory length — all architectures on a shared scale "
                 f"(seed {seed})", fontsize=13, y=1.0)
    fig.tight_layout()
    _save(fig, "fig_trajectories_overlay")
    plt.close(fig)
    print(f"[fin-traj] overlay figure written to {_OUT_DIR / 'fig_trajectories_overlay.png'}")


def make_kd_comparison_figure(seeds: int, widths: List[int]) -> None:
    """Paired bars: 2-PC arc length vs full k-D arc length, per architecture.

    Reads the per-seed CSV (so it needs no re-integration). Faceted width x
    input; within each panel, x = model, two bars per model (2-PC shadow vs full
    k-D path). Makes the projection cost explicit — the k-D bar towers most over
    the 2-PC bar for LTC, i.e. LTC hides the most length in minor components,
    while the architecture ordering is preserved under either metric. Log y
    because LTC dwarfs the others.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    _apply_style()
    per_seed = pd.read_csv(_OUT_DIR / "trajectory_lengths_per_seed.csv")
    per_seed = per_seed[per_seed["finite"]].copy()
    per_seed["model_label"] = per_seed["model"].map(_MODEL_LABELS)
    order = [_MODEL_LABELS[m] for m in _MODEL_ORDER]

    long = per_seed.melt(
        id_vars=["input", "k", "seed", "model_label"],
        value_vars=["arc_length", "arc_length_kd"],
        var_name="metric", value_name="length")
    long["metric"] = long["metric"].map({"arc_length": "2-PC", "arc_length_kd": "full k-D"})
    metric_palette = {"2-PC": "#9DB4C0", "full k-D": "#2F4858"}

    inputs = sorted(long["input"].unique())
    nrows, ncols = len(widths), len(inputs)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 3.8 * nrows),
                             squeeze=False, sharey="row")
    for r, k in enumerate(widths):
        for c, iname in enumerate(inputs):
            ax = axes[r][c]
            sub = long[(long["k"] == k) & (long["input"] == iname)]
            if sub.empty:
                ax.set_axis_off(); continue
            sns.barplot(sub, x="model_label", y="length", hue="metric",
                        order=order, hue_order=["2-PC", "full k-D"],
                        palette=metric_palette, errorbar="sd", capsize=0.1,
                        err_kws={"linewidth": 1.1}, ax=ax)
            ax.set_yscale("log")
            ax.set_xlabel(""); ax.set_ylabel("arc length (log)" if c == 0 else "")
            ax.set_title(f"{iname}  ·  k={k}")
            ax.legend(title="", frameon=False, fontsize=9)
            sns.despine(ax=ax)
    fig.suptitle(f"2-PC shadow vs full k-D trajectory length  ({seeds} seeds, ±sd)",
                 fontsize=13, y=1.0)
    fig.tight_layout()
    _save(fig, "fig_arclength_2pc_vs_kd")
    plt.close(fig)
    print(f"[fin-traj] 2-PC-vs-kD figure written to {_OUT_DIR / 'fig_arclength_2pc_vs_kd.png'}")


def make_figures(seeds: int, widths: List[int], steps: int, fetch: bool) -> None:
    """Publication-quality 2-PC latent-trajectory panels + grouped arc-length bars."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    import seaborn as sns
    from sklearn.decomposition import PCA

    _apply_style()
    per_seed = pd.read_csv(_OUT_DIR / "trajectory_lengths_per_seed.csv")
    cfg = _load_config()

    inputs = {"circle": circular_input(steps)}
    if fetch:
        inputs["financial"] = select_window(materialize_features(cfg), steps)
    input_titles = {"circle": "Synthetic circle", "financial": "AAPL trajectory (2-PC)"}

    # ── Trajectory panels: rows = input type, cols = model (k=max, seed=1) ──────
    # The latent path is drawn with a time-graded colour (viridis) so the reader
    # sees how fast/far each architecture sweeps through state space.
    k_fig = max(widths)
    fig, axes = plt.subplots(len(inputs), 3, figsize=(10.5, 3.4 * len(inputs)),
                             squeeze=False)
    for r, (iname, arr) in enumerate(inputs.items()):
        paths = integrate_models(arr, seed=1, k=k_fig)
        for c, model in enumerate(_MODEL_ORDER):
            proj = PCA(n_components=2).fit_transform(paths[model])
            ax = axes[r][c]
            pts = proj.reshape(-1, 1, 2)
            segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
            lc = LineCollection(segs, cmap="viridis", linewidth=1.8,
                                array=np.linspace(0, 1, len(segs)))
            ax.add_collection(lc)
            ax.scatter(*proj[0], c="#222222", s=28, marker="o", zorder=3)
            ax.scatter(*proj[-1], c="#C44E52", s=46, marker="*", zorder=3)
            ax.autoscale(); ax.set_aspect("equal", "datalim")
            L = arc_length_2pc(paths[model])
            ax.set_title(f"{_MODEL_LABELS[model]}   ·   L = {L:.0f}",
                         color=_MODEL_COLORS[model])
            ax.set_xticks([]); ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(input_titles.get(iname, iname), fontsize=11)
    fig.suptitle(f"Latent trajectory length in 2-PC space  (width k={k_fig}, seed 1)",
                 fontsize=13, y=1.0)
    fig.tight_layout()
    _save(fig, "fig_trajectories")
    plt.close(fig)

    # ── Grouped arc-length bars per width (log y: LTC dwarfs the others) ────────
    per_seed = per_seed[per_seed["finite"]].copy()
    per_seed["model_label"] = per_seed["model"].map(_MODEL_LABELS)
    order = [_MODEL_LABELS[m] for m in _MODEL_ORDER]
    palette = {_MODEL_LABELS[m]: c for m, c in _MODEL_COLORS.items()}
    for k in widths:
        sub = per_seed[per_seed["k"] == k]
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(7, 4.2))
        sns.barplot(sub, x="input", y="arc_length", hue="model_label",
                    hue_order=order, palette=palette, order=sorted(sub["input"].unique()),
                    errorbar="sd", capsize=0.08, err_kws={"linewidth": 1.2}, ax=ax)
        ax.set_yscale("log")
        ax.set_xlabel(""); ax.set_ylabel("2-PC arc length  (log scale)")
        ax.set_title(f"Trajectory length by architecture  (k={k}, {seeds} seeds, ±sd)")
        ax.legend(title="", frameon=False)
        sns.despine(ax=ax)
        fig.tight_layout()
        _save(fig, f"fig_arclength_bar_k{k}")
        plt.close(fig)

    # Shared-scale overlay so the arc-length gap is visible, not autoscaled away.
    make_overlay_figure(widths, steps, fetch)
    # Paired bars exposing how much length the 2-PC projection discards.
    make_kd_comparison_figure(seeds, widths)
    print(f"[fin-traj] figures (PDF + PNG) written to {_OUT_DIR}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=20, help="number of seeds (default 20)")
    ap.add_argument("--widths", type=int, nargs="+", default=[100, 16],
                    help="hidden widths k (default 100 16)")
    ap.add_argument("--steps", type=int, default=100, help="input trajectory length (default 100)")
    ap.add_argument("--no-fetch", action="store_true",
                    help="skip the live financial fetch; run circle baseline only")
    ap.add_argument("--force-refetch", action="store_true",
                    help="ignore the parquet cache and refetch live data")
    ap.add_argument("--no-figures", action="store_true", help="skip figure generation")
    args = ap.parse_args()

    fetch = not args.no_fetch
    summary = run(args.seeds, args.widths, args.steps, fetch=fetch, force=args.force_refetch)

    pd.set_option("display.float_format", lambda v: f"{v:.2f}")
    print("\n=== Trajectory-length summary (mean arc length, finite seeds) ===")
    print("    mean/std = 2-PC arc length;  mean_kd/std_kd = full k-D arc length")
    print(summary[["input", "k", "model_label", "mean", "std", "count",
                   "mean_kd", "std_kd"]].to_string(index=False))

    if not args.no_figures:
        make_figures(args.seeds, args.widths, args.steps, fetch)


if __name__ == "__main__":
    main()
