"""
Directional-skill diagnostic for `cfc_pure_ncp` (run 2a).

Tests whether the CfC's directional accuracy (DA > 0.5) is real skill or just
unconditional drift. Pulls the per-fold prediction snapshots from Supabase
Storage (the `experiment-snapshots` bucket, the same parquets `main.py`
writes), and for each ticker compares the CfC's DA against the *always-long*
base rate — the accuracy of a rule that predicts "up" every day.

Key result (see thesis_results.md Part D): the DA edge over a coin flip is
significant, but the edge over the always-long base rate is ≈0 (binomial
p ≈ 0.5), i.e. no directional skill beyond drift.

Read-only. Writes:
  src/docs/figures/figD_directional_skill.png
  src/docs/figures/data/directional_skill.csv

Run: python scripts/build_directional_skill_figure.py   (repo root, venv active)
"""
from __future__ import annotations

import io
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import dotenv_values
from scipy import stats

HERE = os.path.dirname(__file__)
FIG = os.path.abspath(os.path.join(HERE, "..", "src", "docs", "figures"))
DATA = os.path.join(FIG, "data")
os.makedirs(DATA, exist_ok=True)

RUN2A = "run2a-20260625"
TICKERS = ["AAPL", "SPY", "JPM"]
SEEDS = [42, 43, 44, 45, 46]
MODEL = "cfc_pure_ncp_transformed"
BUCKET = "experiment-snapshots"

# Okabe-Ito colorblind-safe families, matching build_results_figures.py.
C_CFC = "#D55E00"    # CfC directional accuracy
C_BASE = "#999999"   # always-long base rate
C_COIN = "#0072B2"   # coin-flip reference (0.5)


def _safe(token: str) -> str:
    # Mirror main.py::_safe_object_token — '#' etc. map to '_' in Storage keys.
    return re.sub(r"[^A-Za-z0-9._-]", "_", str(token))


def _predictions_path(group: str, ticker: str, run_id: str) -> str:
    return f"{_safe(group)}/{_safe(ticker)}/predictions_{_safe(run_id)}.parquet"


def fetch_predictions() -> pd.DataFrame:
    env = dotenv_values(os.path.join(HERE, "..", ".env"))
    from supabase import create_client
    sb = create_client(env["SUPABASE_URL"], env["SUPABASE_KEY"])
    frames = []
    for tk in TICKERS:
        for s in SEEDS:
            path = _predictions_path(RUN2A, tk, f"{RUN2A}#s{s}")
            data = sb.storage.from_(BUCKET).download(path)
            df = pd.read_parquet(io.BytesIO(data))
            frames.append(df[df.run_key == MODEL])
    return pd.concat(frames, ignore_index=True)


def _assert_seed_agreement(d: pd.DataFrame, tk: str) -> None:
    """The binomial 'one seed of independent dates' rescaling and the paired test
    both assume the 5 seeds are aligned on the *same* dates with identical actuals
    (only `y_pred` may differ, since NN init is seeded). Verify it rather than
    trust it — a missing/failed fold in one seed would silently distort n/k
    (Codex review P2). Keys are (fold_id, t); `y_true` must be seed-invariant."""
    g = d.groupby(["fold_id", "t"])
    n_seeds_per_date = g["seed"].nunique()
    if not (n_seeds_per_date == len(SEEDS)).all():
        bad = int((n_seeds_per_date != len(SEEDS)).sum())
        raise AssertionError(
            f"{tk}: {bad} (fold_id, t) dates do not have all {len(SEEDS)} seeds")
    spread = g["y_true"].agg(lambda s: s.max() - s.min())
    if float(spread.abs().max()) > 1e-9:
        raise AssertionError(f"{tk}: y_true is not identical across seeds")


def analyse(df: pd.DataFrame):
    """Per ticker: pooled DA, the always-long base rate, and TWO tests.

    1. DA vs a coin flip (0.5) — binomial, using one seed's worth of dates as the
       effective sample (the 5 seeds forecast the *same* dates, so pooling would
       inflate N fivefold with no new information; daily 1-step directions are
       near-uncorrelated, so those ~1,735 calls are ~independent Bernoulli trials).
    2. DA vs the always-long baseline — this is a PAIRED comparison, not two
       independent rates (Codex review P2). On days the CfC also predicts "up" it
       *ties* always-long by construction; only its down-call days can win or lose.
       So the honest test is a sign test (exact McNemar) on those discordant
       down-call days: of them, how many did the market actually fall? p ≈ 0.5 (or
       *no* down-call days at all) means the CfC's deviations from always-long add
       nothing. Computed per seed (predictions are seed-specific), pooled for the
       reported n/wins with a per-seed range for transparency.
    """
    rows = []
    for tk in TICKERS:
        d = df[df.ticker == tk]
        _assert_seed_agreement(d, tk)

        yt = d.y_true.values
        yp = d.y_pred.values
        mask = yt != 0.0                      # a zero actual has no direction
        yt_nz, yp_nz = yt[mask], yp[mask]
        da = float(np.mean(np.sign(yt_nz) == np.sign(yp_nz)))
        p_up = float(np.mean(yt_nz > 0.0))
        # For these up-drifting series always-long (predict up every day) is also
        # the best constant-sign rule, since p_up > 0.5 ⇒ p_up = max(p_up, 1-p_up).
        always_long = p_up

        n_seed = int(round(len(yt_nz) / len(SEEDS)))   # one seed's independent dates
        k_half = int(round(da * n_seed))
        p_vs_half = stats.binomtest(k_half, n_seed, 0.5, alternative="greater").pvalue

        # Paired sign test on the CfC's discordant (down-call) days, per seed.
        disc_total = win_total = 0
        per_seed_win = []
        for s in SEEDS:
            ds = d[d.seed == s]
            y, p = ds.y_true.values, ds.y_pred.values
            m = y != 0.0
            y, p = y[m], p[m]
            downcall = np.sign(p) < 0          # CfC deviates from always-long
            n_disc = int(downcall.sum())
            n_win = int((y[downcall] < 0).sum())   # market actually fell → CfC right
            disc_total += n_disc
            win_total += n_win
            if n_disc:
                per_seed_win.append(n_win / n_disc)
        # One-sided sign test: do the down-calls win MORE than half the time?
        p_paired = (stats.binomtest(win_total, disc_total, 0.5,
                                    alternative="greater").pvalue
                    if disc_total else float("nan"))
        downcall_winrate = (win_total / disc_total) if disc_total else float("nan")

        rows.append({
            "ticker": tk, "n_per_seed": n_seed, "cfc_da": da,
            "always_long_base_rate": always_long, "p_up": p_up,
            "edge_vs_coinflip": da - 0.5, "p_binom_vs_0.5": p_vs_half,
            "edge_vs_always_long": da - always_long,
            "downcalls_total_5seeds": disc_total,
            "downcall_win_rate": downcall_winrate,
            "p_paired_sign_test": p_paired,
        })
    return pd.DataFrame(rows)


def backtest(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Long/short strategy on the CfC's predicted sign vs buy-and-hold, in the
    model's transformed target units, for a single seed. Reproduces the sign-flip
    and gross-return numbers cited in thesis_results.md Part D-DA (Codex review P2:
    those numbers must come from a committed artifact, not an ad-hoc scratch run).
    Costs are omitted because they only ever *reduce* the strategy return, so
    'strategy ≤ buy-and-hold gross' already settles the question."""
    rows = []
    for tk in TICKERS:
        d = df[(df.ticker == tk) & (df.seed == seed)].sort_values(["fold_id", "t"])
        yt = d.y_true.values
        pos = np.sign(d.y_pred.values)
        pos[pos == 0] = 1.0                   # flat prediction → hold long
        flips = int(np.sum(np.abs(np.diff(pos)) > 0))
        yrs = len(yt) / 252.0
        rows.append({
            "ticker": tk, "seed": seed,
            "strat_gross": float(np.sum(pos * yt)),
            "buy_and_hold": float(np.sum(yt)),
            "flips_per_year": flips / yrs,
        })
    return pd.DataFrame(rows)


def make_figure(res: pd.DataFrame, path: str):
    x = np.arange(len(res))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7.2, 4.2))

    ax.bar(x - w / 2, res.cfc_da, w, color=C_CFC, label="cfc_pure_ncp DA", zorder=3)
    ax.bar(x + w / 2, res.always_long_base_rate, w, color=C_BASE,
           label="always-long base rate", zorder=3)
    ax.axhline(0.5, color=C_COIN, ls="--", lw=1.4, zorder=2,
               label="coin flip (0.5)")

    for xi, r in zip(x, res.itertuples()):
        ax.text(xi - w / 2, r.cfc_da + 0.004, f"{r.cfc_da:.3f}",
                ha="center", va="bottom", fontsize=8.5, color=C_CFC)
        ax.text(xi + w / 2, r.always_long_base_rate + 0.004,
                f"{r.always_long_base_rate:.3f}", ha="center", va="bottom",
                fontsize=8.5, color="#555555")
        # Honest paired test: sign test on the CfC's down-call days. NaN means the
        # CfC made *no* down-calls, so it equals always-long by construction.
        note = ("no down-calls" if np.isnan(r.p_paired_sign_test)
                else f"sign p={r.p_paired_sign_test:.2f}")
        ax.text(xi, 0.508, note, ha="center", va="bottom", fontsize=8,
                color="#333333")

    ax.set_xticks(x)
    ax.set_xticklabels(res.ticker)
    ax.set_ylim(0.495, max(res.always_long_base_rate.max(), res.cfc_da.max()) + 0.02)
    ax.set_ylabel("directional accuracy")
    ax.set_title("CfC directional accuracy vs the always-long base rate (run 2a)\n"
                 "edge over drift ≈ 0 → no directional skill", fontsize=10.5)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#e6e6e6", zorder=0)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"wrote {path}")


if __name__ == "__main__":
    print("fetching cfc_pure_ncp prediction snapshots ...")
    df = fetch_predictions()

    res = analyse(df)
    csv = os.path.join(DATA, "directional_skill.csv")
    res.to_csv(csv, index=False, float_format="%.6f")
    print(f"wrote {csv}")
    print(res.to_string(index=False))

    bt = backtest(df)
    bt_csv = os.path.join(DATA, "directional_skill_backtest.csv")
    bt.to_csv(bt_csv, index=False, float_format="%.6f")
    print(f"wrote {bt_csv}")
    print(bt.to_string(index=False))

    make_figure(res, os.path.join(FIG, "figD_directional_skill.png"))
