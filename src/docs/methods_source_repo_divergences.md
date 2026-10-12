# Methods divergences: this grid vs. the LTC / CfC source repositories

How the forecasting grid (`grid-2026-06-11`, AAPL/SPY/JPM log-return prediction) relates
to the original authors' own experiments. The neural models are built with **Mathias
Lechner's `ncps` library** (v0.0.2) — a later package that *adds* Neural-Circuit-Policy
(NCP) wiring support to LTC/CfC — not the paper repositories directly. The two upstream
repos are vendored (code only) under `vendor/`:

- `vendor/ltc_reference/` — Hasani et al., *Liquid Time-constant Networks* (AAAI 2021),
  `github.com/raminmh/liquid_time_constant_networks` (Apache-2.0).
- `vendor/cfc_reference/` — Hasani, Lechner et al., *Closed-form Continuous-time Neural
  Networks* (Nature MI 2022), `github.com/raminmh/CfC` (Apache-2.0).

Nothing below is invalid; every item is a **deliberate** choice. The point is that the
grid does **not** replicate the papers' benchmark setups, so the methods section should
own these as design decisions rather than imply a reproduction.

## 1. NCP wiring vs. fully-connected (the main one)
- **LTC paper** (`experiments_with_ltcs/ltc_model.py`): `LTCCell(num_units)` is **fully
  connected**, default size **32**, `ode_unfolds = 6`, affine input mapping.
- **CfC paper** (`traffic_with_cfc.py:123-129`): `CfcCell(units)` with a **dense MLP
  backbone** (`backbone_units` 128-256, 1 layer, silu/lecun, dropout). No NCP.
- **This grid** (`src/models/ltc_model.py:440`, `src/models/cfc_model.py`):
  `AutoNCP(units=16, output_size=1, sparsity_level=0.5)` — **sparse NCP wiring**. With
  `output_size=1` there is a single motor neuron, so the effective recurrent core is
  smaller than "16 dense." Neither benchmark paper used NCP wiring (NCP is a separate
  line: Lechner et al. 2020, *Neural Circuit Policies*, Nature MI).

## 2. No backbone (CfC)
Their CfC sits behind a 128-256-unit MLP backbone; the grid's AutoNCP CfC has none.

## 3. Capacity
Theirs: 32 dense (LTC) / 128-256 backbone (CfC). Grid: 16-unit AutoNCP. Much smaller —
on-thesis ("compact, expressive"), but a divergence, not a match.

## 4. Evaluation split — the grid is *stricter*, not looser
LTC `traffic.py:60-73` random-permutes **overlapping** sliding windows into valid (10%) /
test (15%) / train — which **leaks** (adjacent overlapping windows can land in both train
and test). The grid uses **walk-forward, no lookahead**. So lower grid numbers partly
reflect removing a leak the original eval had. State this as a strength.

## 5. Training budget / model selection
LTC paper: **200 epochs + best-validation checkpoint** (`fit()` tracks `best_valid_loss`,
restores best → de-facto early stopping). Grid: **fixed 50 epochs, no early stopping**
(deliberate cross-model fairness). The grid's 50 epochs matches the **CfC** repo configs
(`epochs: 50`), just not LTC's 200 + best-val.

## 6. Task domain
Their benchmarks (traffic volume, gestures, occupancy) carry real signal. The grid is
**low-SNR multi-step log-return forecasting** — outside their domain. The grid's
near-coin-flip directional accuracy therefore does **not** contradict the papers; they
never claimed LTC/CfC forecast near-random financial returns.

## Consequences for the write-up
- **Do not compare grid numbers to the papers' benchmark numbers** — different
  architecture (wiring / size / backbone), different split, different task.
- **Expressivity-bench consistency:** `src/analysis/hasani_trajectory.py` (the faithful
  trajectory-length replication) uses the papers' **fully-connected** cells, while the grid
  uses **NCP-16** cells. These are two different LTCs. Keep the link explicit: the
  expressivity claim is about the *dynamics class*, not the specific grid model.

## Turning this into a result (optional Run 2 ablation)
Every variant is a one-line change in the `ncps` builders (verified, v0.0.2):
`units=AutoNCP(...)` (NCP) vs `units=16` (int → fully-connected); `mode ∈
{default, pure, no_gate}`; `mixed_memory=True`; `backbone_units` / `backbone_layers`.
A CfC / wiring ablation on the same financial pipeline turns these divergences into a
robustness statement — "the null forecasting result holds across CfC variants,
NCP-vs-dense wiring, backbone, and capacity." Pre-req: parameterize `cfc_model.py` /
`ltc_model.py` to read `mode` / `mixed_memory` / `backbone` / wiring-type from
`config.yaml` (currently hardcoded `AutoNCP` + `default`).
