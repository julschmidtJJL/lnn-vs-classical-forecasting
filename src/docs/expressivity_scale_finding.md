# Expressivity "open the black box" chapter — the scale-artifact finding

Status: analysis complete (run-#1/#2-independent half). 2026-06-18.
Code: `src/analysis/run_expressivity_chapter.py`, `trajectory_length.trajectory_length_normalized`.
Tests: `tests/test_trajectory_length.py`, `tests/test_expressivity_chapter.py` (green).

## What this chapter does
The forecasting comparison (run #0/#1/#2) is the thesis setup: classical AR(1)/SARIMAX beats every
neural model on daily log-returns. This chapter opens the black box — it asks whether the *theoretical*
LNN advantage (Hasani's expressivity claim, measured as latent trajectory length) is real on financial
data, using Hasani's own apparatus, at random init (their exact protocol).

Two tracks, two input protocols, reported separately:
- **Canonical** — the faithful Hasani cells (Neural-ODE, CT-RNN, LTC) sharing weights/width, fed a
  financial input reduced to m=2 PCs (the documented stiffness regime). Reports raw 2-PC arc length
  (Hasani parity) AND scale-invariant `L_norm`.
- **Bench** — the production architectures (LTC, CfC, CTRNN, LSTM) at init, fed the full standardized
  feature window. Reports `L_norm` (the expressivity x-axis the forecast-skill bridge joins later).

`L_norm = arc_length(proj) / ((T-1) * RMS_radius(proj))` in 2-PC space — scale-invariant, so it
compares path complexity across architectures on different latent scales (raw arc length does not:
a larger hidden-state norm inflates the path, a scale artifact, not expressivity).

## Headline finding: the LTC dominance is largely a SCALE artifact
Canonical track, AAPL, 25 non-overlapping windows × 10 seeds, k=100:

| metric | LTC/CT-RNN | LTC/Neural-ODE |
|---|---|---|
| RAW arc length (replicates Hasani) | 12.6× | 5.9× |
| scale-invariant L_norm | **1.00×** | **~1.0×** |

It is NOT a finance quirk — same on Hasani's OWN synthetic circle (k=100, 20 seeds):
raw LTC/CT-RNN 13.8×, LTC/Neural-ODE 8.8× → L_norm 1.08× and 1.62×.

So the order-of-magnitude dominance is mostly latent **magnitude**, not per-radius geometric
complexity. LTC vs CT-RNN is *entirely* scale on both inputs; LTC vs Neural-ODE keeps a modest
scale-free edge (≤1.6× on the circle), erased to ~1.0× on noisy financial input.

`L_norm` is not a wash-out (it has discriminating power): the bench track keeps real spread —
CfC 0.32 > LTC 0.27 > CT-RNN 0.22 > LSTM 0.20. Liquid nets retain a modest scale-free edge over LSTM.

## Robustness — m-sensitivity (AAPL canonical, k=100, 3 seeds × 25 windows)
| m | %var | finite | RAW LTC/CT | RAW LTC/ND | L_norm LTC/CT | L_norm LTC/ND |
|---|---|---|---|---|---|---|
| 2 | 48% | 100% | 12.7× | 6.1× | 0.99× | 1.02× |
| 4 | 72% | 100% | 13.2× | 6.4× | 1.00× | 1.17× |
| 8 | 90% | 100% | 14.0× | 7.3× | 1.06× | 1.23× |

- The collapse is NOT an artifact of the lossy m=2 shadow — it holds at m=8 (90% variance);
  L_norm LTC/CT-RNN stays ~1.0× at every m.
- LTC's scale-free edge over Neural-ODE grows modestly with input richness (1.02→1.23×) but never
  approaches the raw 6–7×.
- Financial input does NOT go stiff up to m=8 (100% finite) — so m=2 is a Hasani-parity choice here,
  not a numerical necessity.

## Framing (decided): report both; the divergence is the content
Do NOT declare the dominance "fake" — magnitude-as-expressivity is a defensible rebuttal
(supervisor is dynamical-systems). Present raw (replicates Hasani) AND L_norm (parity / modest edge),
let the divergence be the finding. This strengthens the negative-result thesis: the expressivity
advantage is partly illusory (scale) AND does not transfer to forecast skill.

## Methods caveats to state explicitly
- The canonical input "two m's" are PC1 (~32% var) and PC2 (~18%) of the per-window standardized
  features — together ~48%, a lossy 2-D shadow. The bench track uses all 487 features.
- All measurements at random init (Hasani's trajectory-length protocol).

## Not done yet (deferred)
- Forecast-skill axis + the expressivity↔skill bridge → needs run #1/#2 (the y-axis).
- SPY/JPM canonical/bench (needs live feature fetch) — 3-ticker robustness.
- Circle raw-vs-L_norm as a chapter figure; noise control (T4); regime stability (T5).
