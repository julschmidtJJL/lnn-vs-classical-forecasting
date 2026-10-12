"""
Validation gate + extraction tests for the trajectory-length bench.

The gate is the de-risk step: it asserts the trajectory-length MEASURE behaves
correctly before it is trusted on financial data. Measured AT INITIALISATION
(random weights, fixed seeds) because trajectory-length expressivity bounds
(Raghu 2017; Hasani Thm 4/5) are an init-time property -- this makes the check
deterministic, not flaky. Primary assertion: L rises monotonically with network
width (reproduces Hasani Fig 2C/3B), Spearman rho >= 0.9.

These tests build small torch models, so they are heavier than the pure-core
unit tests; kept in their own file.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

torch = pytest.importorskip("torch")
pytest.importorskip("ncps")
spstats = pytest.importorskip("scipy.stats")

from src.analysis.expressivity_extract import build_bench_model, extract_latent
from src.analysis.trajectory_length import trajectory_length


def _synthetic_sequences(n=30, seq_len=15, n_feat=6, seed=0):
    """n sequences of 3 random-phase sinusoids + light noise (Hasani-style
    controlled input). Deterministic given seed."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 2 * np.pi, seq_len)
    out = []
    for _ in range(n):
        s = np.zeros((seq_len, n_feat))
        for f in range(n_feat):
            for _k in range(3):
                s[:, f] += np.sin(t * rng.uniform(0.5, 3.0) + rng.uniform(0, 2 * np.pi))
        s += rng.normal(0, 0.1, (seq_len, n_feat))
        out.append(s)
    return out


# ─── extraction shape contracts ──────────────────────────────────────────────

@pytest.mark.parametrize("arch", ["ltc", "cfc"])
def test_extract_latent_ct_shape(arch):
    m = build_bench_model(arch, units=16, n_features=6, seed=0)
    lat = extract_latent(m, np.zeros((15, 6)), arch)
    assert lat.shape == (15, 16)  # full NCP neuron state, not the 1-dim output


def test_extract_latent_lstm_shape():
    lstm = torch.nn.LSTM(input_size=6, hidden_size=12, num_layers=2, batch_first=True)
    lat = extract_latent(lstm, np.zeros((15, 6)), "lstm")
    assert lat.shape == (15, 12)  # top-layer per-timestep hidden


def test_extract_latent_production_ctrnn_shape():
    m = build_bench_model("ctrnn", units=16, n_features=6, seed=0)
    lat = extract_latent(m, np.zeros((15, 6)), "ctrnn")
    assert lat.shape == (15, 16)  # per-step hidden via return_states


def test_extract_latent_rejects_non_2d():
    m = build_bench_model("ltc", units=16, n_features=6, seed=0)
    with pytest.raises(ValueError):
        extract_latent(m, np.zeros((15, 6, 1)), "ltc")


# ─── the validation gate ─────────────────────────────────────────────────────

@pytest.mark.parametrize("arch", ["ltc", "cfc"])
def test_validation_gate_width_monotonicity(arch):
    """L rises monotonically with width at init (Spearman rho >= 0.9), averaged
    over init seeds. Observed rho = 1.000 for both archs at authoring; 0.9 is the
    contract threshold with margin."""
    widths = [8, 16, 32, 64]
    n_feat = 6
    seqs = _synthetic_sequences(n=30, seq_len=15, n_feat=n_feat, seed=0)
    width_means = []
    for w in widths:
        per_seed = []
        for seed in range(4):
            m = build_bench_model(arch, w, n_feat, seed)
            Ls = [trajectory_length(extract_latent(m, s, arch))[0] for s in seqs]
            per_seed.append(float(np.mean(Ls)))
        width_means.append(float(np.mean(per_seed)))
    rho, _ = spstats.spearmanr(widths, width_means)
    assert rho >= 0.9, f"{arch}: width-monotonicity broke, rho={rho:.3f}, means={width_means}"
