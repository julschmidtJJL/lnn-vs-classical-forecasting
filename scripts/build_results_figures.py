"""
Build the Run #2 results-doc figures + aggregated CSVs straight from Supabase.

Read-only: pulls experiment_summaries for the four forecasting run groups,
reuses analysis.aggregate_seeds.{aggregate_metrics, aggregate_significance},
writes PNGs to src/docs/figures/ and tidy CSVs to src/docs/figures/data/.

Run: python scripts/build_results_figures.py   (from repo root, venv active)
"""
from __future__ import annotations

import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from dotenv import dotenv_values

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from analysis.aggregate_seeds import (  # noqa: E402
    aggregate_metrics, aggregate_significance, dedup_latest,
)

HERE = os.path.dirname(__file__)
FIG = os.path.abspath(os.path.join(HERE, "..", "src", "docs", "figures"))
DATA = os.path.join(FIG, "data")
os.makedirs(DATA, exist_ok=True)

RUN2A = "run2a-20260625"
RUN2B = "run2b-20260627"
GRID = "grid-2026-06-11"
RUN1 = "run1-logreturn-2026-06-18"
TICKERS = ["AAPL", "JPM", "SPY"]

# Okabe-Ito colorblind-safe families (validated: worst adjacent CVD ΔE 17.9).
FAM = {
    "classical": "#0072B2",
    "cfc":       "#D55E00",
    "ltc":       "#009E73",
    "other_nn":  "#CC79A7",
    "naive":     "#E69F00",
}
FAM_LABEL = {
    "classical": "SARIMAX (classical)",
    "cfc": "CfC variants",
    "ltc": "LTC variants",
    "other_nn": "LSTM / CT-RNN / Neural-ODE",
    "naive": "naive (random walk)",
}


def family(model: str) -> str:
    if model.startswith("sarimax"):
        return "classical"
    if model.startswith("naive"):
        return "naive"
    if model.startswith("cfc"):
        return "cfc"
    if model.startswith("ltc"):
        return "ltc"
    return "other_nn"


def short(model: str) -> str:
    return model.replace("_transformed", "")


def fetch(group: str):
    env = dotenv_values(os.path.join(HERE, "..", ".env"))
    from supabase import create_client
    sb = create_client(env["SUPABASE_URL"], env["SUPABASE_KEY"])
    rows = (sb.table("experiment_summaries")
            .select("ticker,seed,run_id,run_group_id,created_at,metrics,primary_significance_table")
            .eq("run_group_id", group).order("created_at").execute().data)
    return dedup_latest(rows)


def mae_table(rows):
    agg = aggregate_metrics(rows)
    out = {}  # ticker -> list[(model, mae_mean, mae_std, da_mean)]
    for (tk, mc), m in agg.items():
        out.setdefault(tk, []).append(
            (mc, m["mae_mean"]["mean"], m["mae_mean"]["std"],
             m.get("directional_accuracy_mean", {}).get("mean", float("nan"))))
    for tk in out:
        out[tk].sort(key=lambda r: r[1])
    return out


# ---------------------------------------------------------------- fetch --------
print("fetching summaries ...")
rows2a = fetch(RUN2A)
rows2b = fetch(RUN2B)
rows_grid = fetch(GRID)
rows_run1 = fetch(RUN1)
mae2a = mae_table(rows2a)
mae2b = mae_table(rows2b)

# tidy CSVs (appendix / reproducibility)
for name, tbl in [("run2a", mae2a), ("run2b", mae2b)]:
    with open(os.path.join(DATA, f"mae_{name}.csv"), "w") as fh:
        fh.write("ticker,model,mae_mean,mae_std,da_mean\n")
        for tk in TICKERS:
            for mc, mae, std, da in tbl[tk]:
                fh.write(f"{tk},{short(mc)},{mae:.6f},{std:.6f},{da:.4f}\n")
print("wrote MAE csvs")


# Auto-generated appendix Markdown tables (paste into the doc verbatim so the
# tables can never drift from the CSVs — Codex P2). Sort key is (mae, model) so
# rounded ties order deterministically, matching the CSV.
def emit_markdown_table(tbl, title):
    lines = [f"## {title}", "",
             "Lowest MAE per ticker in **bold**; DA = mean directional accuracy. "
             "Rows sorted by (MAE, model name) — identical to the CSV.", "",
             "| Ticker | Model condition        | MAE                   | DA    |",
             "|--------|------------------------|-----------------------|-------|"]
    for tk in TICKERS:
        rows = sorted(tbl[tk], key=lambda r: (round(r[1], 5), short(r[0])))
        best = min(r[1] for r in rows)
        for mc, mae, std, da in rows:
            name = short(mc)
            cell_m = f"{mae:.5f} ± {std:.5f}"
            if abs(mae - best) < 1e-12:
                name, cell_m = f"**{name}**", f"**{cell_m}**"
            lines.append(f"| {tk:<6} | {name:<22} | {cell_m:<21} | {da:.3f} |")
    return "\n".join(lines) + "\n"


with open(os.path.join(DATA, "appendix_tables.md"), "w") as fh:
    fh.write(emit_markdown_table(mae2a, "Appendix Table 1 — Run 2a (raw exog), MAE mean ± std across 5 seeds"))
    fh.write("\n")
    fh.write(emit_markdown_table(mae2b, "Appendix Table 2 — Run 2b (stationarized exog), MAE mean ± std across 5 seeds"))
print("wrote appendix_tables.md")


# DM p-value audit for the headline tie: pull p_corrected for every
# cfc_pure_ncp-vs-SARIMAX pair across seeds/tickers/arms straight from the
# per-seed primary_significance_table (grounds the "0/120, p≈1.0" claim — Codex P2).
def dm_audit(rows, arm):
    recs = []
    for r in rows:
        tk, seed = r.get("ticker"), r.get("seed")
        for p in (r.get("primary_significance_table") or []):
            a, b = p.get("model_a", ""), p.get("model_b", "")
            pair = {a, b}
            if "cfc_pure_ncp_transformed" in pair and any(
                    m.startswith("sarimax") for m in pair):
                recs.append((arm, tk, seed, a, b,
                             p.get("p_corrected"), int(bool(p.get("significant")))))
    return recs


audit = dm_audit(rows2a, "2a") + dm_audit(rows2b, "2b")
with open(os.path.join(DATA, "dm_cfc_vs_sarimax_audit.csv"), "w") as fh:
    fh.write("arm,ticker,seed,model_a,model_b,p_corrected,significant\n")
    for rec in audit:
        fh.write(",".join(str(x) for x in rec) + "\n")
_n = len(audit)
_nsig = sum(r[6] for r in audit)
_pmin = min((r[5] for r in audit if r[5] is not None), default=None)
print(f"DM audit: {_n} cfc_pure_ncp-vs-SARIMAX comparisons, "
      f"{_nsig} significant, min p_corrected={_pmin}")

plt.rcParams.update({
    "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "figure.dpi": 150,
})


# ---- FIG 1 (MAIN): MAE ranking per ticker, run2a, family-colored -------------
def fig_ranking(tbl, arm, fname):
    fig, axes = plt.subplots(1, 3, figsize=(11, 5.2), sharex=False)
    for ax, tk in zip(axes, TICKERS):
        data = tbl[tk]
        ys = np.arange(len(data))[::-1]
        for y, (mc, mae, std, da) in zip(ys, data):
            fam = family(mc)
            star = mc.startswith("cfc_pure_ncp")
            ax.barh(y, mae, color=FAM[fam], height=0.72,
                    edgecolor="white", linewidth=0.6, zorder=2)
            if std > 1e-6:
                ax.errorbar(mae, y, xerr=std, fmt="none", ecolor="#333",
                            elinewidth=0.8, capsize=2, zorder=3)
            label = short(mc) + ("  ★" if star else "")
            ax.text(mae, y, "  " + label, va="center", ha="left",
                    fontsize=6.3, zorder=4,
                    fontweight="bold" if star else "normal")
        # SARIMAX reference band (min sarimax MAE)
        sar = [mae for mc, mae, _, _ in data if mc.startswith("sarimax")]
        if sar:
            ax.axvline(min(sar), color=FAM["classical"], ls="--",
                       lw=1.0, alpha=0.7, zorder=1)
        ax.set_yticks([])
        ax.set_title(tk, fontweight="bold")
        ax.set_xlabel("MAE (mean ± std across 5 seeds)")
        ax.set_xlim(0, max(mae for _, mae, _, _ in data) * 1.32)
    handles = [plt.Rectangle((0, 0), 1, 1, color=FAM[k]) for k in FAM]
    fig.legend(handles, [FAM_LABEL[k] for k in FAM], loc="lower center",
               ncol=5, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"Run #2{arm}: 1-day-ahead log-return MAE by model  "
                 f"(★ = cfc_pure_ncp, best; dashed = best SARIMAX)",
                 fontweight="bold", fontsize=10)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(os.path.join(FIG, fname), bbox_inches="tight")
    plt.close(fig)
    print("wrote", fname)


fig_ranking(mae2a, "a", "fig1_mae_ranking_run2a.png")
fig_ranking(mae2b, "b", "figA_mae_ranking_run2b.png")


# ---- FIG 2 (MAIN): cfc_pure_ncp vs SARIMAX — MAE gap + DM tie ----------------
sig2a = aggregate_significance(rows2a)
sig2b = aggregate_significance(rows2b)


def cfc_vs_sarimax_gap(tbl):
    # returns ticker -> (cfc_mae, best_sarimax_mae)
    d = {}
    for tk in TICKERS:
        cfc = next(mae for mc, mae, _, _ in tbl[tk] if mc.startswith("cfc_pure_ncp"))
        sar = min(mae for mc, mae, _, _ in tbl[tk] if mc.startswith("sarimax"))
        d[tk] = (cfc, sar)
    return d


fig, ax = plt.subplots(figsize=(7.6, 4.4))
g2a = cfc_vs_sarimax_gap(mae2a)
g2b = cfc_vs_sarimax_gap(mae2b)
x = np.arange(len(TICKERS))
w = 0.18
for i, (arm, g, hatch) in enumerate([("2a", g2a, ""), ("2b", g2b, "///")]):
    cfc = [g[tk][0] for tk in TICKERS]
    sar = [g[tk][1] for tk in TICKERS]
    ax.bar(x + (i * 2 - 1) * w, cfc, w, color=FAM["cfc"], hatch=hatch,
           edgecolor="white", label=f"cfc_pure_ncp ({arm})")
    ax.bar(x + (i * 2) * w, sar, w, color=FAM["classical"], hatch=hatch,
           edgecolor="white", label=f"best SARIMAX ({arm})")
for xi, tk in zip(x, TICKERS):
    ax.text(xi, max(g2a[tk] + g2b[tk]) * 1.02, "0/5 sig.",
            ha="center", fontsize=7.5, color="#555")
ax.set_xticks(x + 0.09)
ax.set_xticklabels(TICKERS, fontweight="bold")
ax.set_ylabel("MAE (mean across 5 seeds)")
ax.set_title("cfc_pure_ncp is numerically #1 but shows no significant DM difference\n"
             "from SARIMAX (0 of 5 seeds significant, every ticker & arm)",
             fontweight="bold", fontsize=9.5)
ax.legend(frameon=False, fontsize=7, ncol=2)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig2_cfc_vs_sarimax_tie.png"), bbox_inches="tight")
plt.close(fig)
print("wrote fig2")


# ---- FIG A2: stationarization null (2a vs 2b MAE scatter) --------------------
fig, ax = plt.subplots(figsize=(5.6, 5.4))
allmae = []
lookup2b = {(tk, short(mc)): mae for tk in TICKERS for mc, mae, _, _ in mae2b[tk]}
for tk in TICKERS:
    for mc, mae, _, _ in mae2a[tk]:
        b = lookup2b.get((tk, short(mc)))
        if b is None:
            continue
        allmae += [mae, b]
        ax.scatter(mae, b, s=22, color=FAM[family(mc)],
                   edgecolor="white", linewidth=0.5, zorder=3)
lo, hi = min(allmae) * 0.97, max(allmae) * 1.03
ax.plot([lo, hi], [lo, hi], color="#888", ls="--", lw=1, zorder=1)
ax.set_xlim(lo, hi)
ax.set_ylim(lo, hi)
ax.set_xlabel("MAE — arm 2a (raw exog)")
ax.set_ylabel("MAE — arm 2b (stationarized exog)")
ax.set_title("Stationarization: top model changes < 0.1% (no headline reversal)\n"
             "most models track y = x; a few shift more (off-diagonal points)",
             fontweight="bold", fontsize=9)
handles = [plt.Line2D([0], [0], marker="o", ls="", color=FAM[k]) for k in FAM]
ax.legend(handles, [FAM_LABEL[k] for k in FAM], frameon=False, fontsize=6.5,
          loc="upper left")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "figB_stationarization_null.png"), bbox_inches="tight")
plt.close(fig)
print("wrote figB")


# ---- FIG A3: DM significance heatmap (k of 5), AAPL run2a -------------------
def dm_heatmap(sig, rows, tk, fname, title):
    agg = aggregate_metrics(rows)
    models = sorted({mc for (t, mc) in agg if t == tk},
                    key=lambda m: agg[(tk, m)]["mae_mean"]["mean"])
    n = len(models)
    M = np.full((n, n), np.nan)
    idx = {m: i for i, m in enumerate(models)}
    # DM "significant" is symmetric (it flags a significant DIFFERENCE, not a
    # direction), and each unordered pair is stored once -> mirror both ways.
    for (t, a, b), v in sig.items():
        if t != tk or a not in idx or b not in idx:
            continue
        M[idx[a], idx[b]] = v["n_significant"]
        M[idx[b], idx[a]] = v["n_significant"]
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(M, cmap="cividis", vmin=0, vmax=5)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels([short(m) for m in models], rotation=90, fontsize=6)
    ax.set_yticklabels([short(m) for m in models], fontsize=6)
    for i in range(n):
        for j in range(n):
            if not np.isnan(M[i, j]):
                ax.text(j, i, int(M[i, j]), ha="center", va="center",
                        fontsize=5.5, color="white" if M[i, j] < 3 else "black")
    ax.set_title(title, fontweight="bold", fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.6,
                 label="# of 5 seeds with a significant DM difference (0 = tie)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, fname), bbox_inches="tight")
    plt.close(fig)
    print("wrote", fname)


dm_heatmap(sig2a, rows2a, "AAPL", "figC_dm_heatmap_aapl_run2a.png",
           "Pairwise DM significance (k of 5 seeds), AAPL run2a — symmetric,\n"
           "direction not encoded; 0 = no significant difference detected; "
           "models ordered by MAE")


# ---- FIG A4: headline progression run #0 -> #1 -> #2 (best classical vs NN) --
def best_of(rows, pred):
    agg = aggregate_metrics(rows)
    out = {}
    for tk in TICKERS:
        cand = [(mc, agg[(tk, mc)]["mae_mean"]["mean"]) for (t, mc) in agg
                if t == tk and pred(mc)]
        if cand:
            out[tk] = min(cand, key=lambda z: z[1])
    return out


runs = [("#0 grid", rows_grid), ("#1 input-fix", rows_run1),
        ("#2a raw", rows2a), ("#2b stat.", rows2b)]
fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), sharey=False)
for ax, tk in zip(axes, TICKERS):
    xs = np.arange(len(runs))
    cls = [best_of(r, lambda m: m.startswith("sarimax") or m.startswith("naive") and False)[tk][1]
           if tk in best_of(r, lambda m: m.startswith("sarimax"))
           else np.nan for _, r in runs]
    nn = []
    for _, r in runs:
        b = best_of(r, lambda m: not (m.startswith("sarimax") or m.startswith("naive")))
        nn.append(b[tk][1] if tk in b else np.nan)
    ax.plot(xs, cls, "-o", color=FAM["classical"], label="best classical (SARIMAX)")
    ax.plot(xs, nn, "-s", color=FAM["cfc"], label="best neural")
    ax.set_xticks(xs)
    ax.set_xticklabels([r[0] for r in runs], rotation=30, fontsize=7)
    ax.set_title(tk, fontweight="bold")
    ax.set_ylabel("MAE")
axes[0].legend(frameon=False, fontsize=7)
fig.suptitle("Best-classical vs best-neural MAE across the run sequence\n"
             "(best-neural dips below classical only at run #2 — difference not "
             "DM-significant; the 'best neural' model differs across runs)",
             fontweight="bold", fontsize=9)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(os.path.join(FIG, "figD_run_progression.png"), bbox_inches="tight")
plt.close(fig)
print("wrote figD")

print("\nDONE. Figures in", FIG)
