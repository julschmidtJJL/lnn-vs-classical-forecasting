"""
Per-architecture latent extraction for the trajectory-length bench.

Kept separate from trajectory_length.py so the numeric core stays torch-free.
``extract_latent`` returns the latent state path a model traces over one input
sequence, shape (T, d), which trajectory_length.trajectory_length() then reduces
to a scalar arc length.

All models are measured on a SHARED input-time axis (one latent point per input
timestep), so their arc lengths are directly comparable.

This module hosts the PRODUCTION-model track (`build_bench_model`): the actual
grid architectures (NCP-wired LTC/CfC, the production CT-RNN, a plain LSTM), for
a "do our real models behave the same way?" measurement -- chiefly the
width-monotonicity gate (L rises with width at init; Raghu 2017 / Hasani
Thm 4/5). The faithful cross-model expressivity ORDERING (Hasani Table 2) lives
in the standalone port ``hasani_trajectory.py``, which superseded the earlier
in-module canonical common-form cells.

Latent definitions (verified against src/models/*.py and the installed ncps API):
  - ncps ltc/cfc : full NCP neuron state, by stepping ``module.rnn_cell`` over the
    sequence. ncps ``return_sequences=True`` only exposes the 1-dim motor output,
    not the latent, so we step the cell directly for the full (units)-dim state.
  - lstm         : top-layer per-timestep hidden state (the nn.LSTM ``output``).
  - production CT-RNN : per-timestep hidden path via the additive
    ``return_states=True`` path on the module's forward.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def _finite_or_raise(arr: np.ndarray, arch: str) -> np.ndarray:
    """Guard against silent NaN/inf latents (a random-init Neural ODE can
    diverge). Callers (the runner) catch this, log, and skip the sample rather
    than letting a non-finite arc length poison the aggregate.
    """
    if not np.isfinite(arr).all():
        raise FloatingPointError(
            f"extract_latent('{arch}') produced non-finite latents "
            f"({int((~np.isfinite(arr)).sum())} bad values)."
        )
    return arr


def extract_latent(module: torch.nn.Module, X, arch: str) -> np.ndarray:
    """Latent state path of ``module`` over one input sequence ``X`` (T, n_feat).

    Returns a float64 ndarray of shape (T, d). Dispatch is on the module's
    capability, not the ``arch`` string, so the same arch name works whether it
    is a canonical cell, a production net, or an ncps module. Runs in eval /
    no-grad; does not mutate the module beyond toggling eval(). Raises
    FloatingPointError if the latent path is non-finite.
    """
    arch = arch.lower()
    xt = torch.as_tensor(np.asarray(X), dtype=torch.float32)
    if xt.ndim != 2:
        raise ValueError(f"X must be 2-D (T, n_features); got {tuple(xt.shape)}")
    T = xt.shape[0]
    module.eval()
    with torch.no_grad():
        if hasattr(module, "rnn_cell"):                      # ncps LTC / CfC
            cell = module.rnn_cell
            h = torch.zeros(1, cell.state_size)
            states = []
            for t in range(T):
                _out, h = cell.forward(xt[t : t + 1], h, 1.0)  # positional elapsed_time
                states.append(h[0].clone())
            latent = torch.stack(states)
        elif isinstance(module, torch.nn.LSTM):
            out, _ = module(xt.unsqueeze(0))                 # (1, T, hidden)
            latent = out[0]
        elif hasattr(module, "lstm") and isinstance(module.lstm, torch.nn.LSTM):
            out, _ = module.lstm(xt.unsqueeze(0))
            latent = out[0]
        else:
            # Generic per-step CT contract: forward(x, return_states=True) ->
            # (1, T, hidden). Covers the production CT-RNN.
            H = module(xt.unsqueeze(0), return_states=True)
            latent = H[0]
    arr = latent.cpu().numpy().astype(np.float64)
    return _finite_or_raise(arr, arch)


# ─── Production thesis models (separate "do our real models agree?" track) ────

def build_bench_model(arch: str, units: int, n_features: int, seed: int) -> torch.nn.Module:
    """Fresh, deterministically-initialised PRODUCTION model for ``arch``:
    ``ltc``/``cfc`` are NCP (AutoNCP) wired; ``ctrnn`` is the production
    ``_CTRNNNetwork``; ``lstm`` is a plain single-layer ``nn.LSTM``. (The grid's
    Neural-ODE is a GRU-encoder forecaster, not measurable on the shared
    input-time axis; the canonical ``neural_ode`` cell stands in for it.)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    arch = arch.lower()
    if arch in ("ltc", "cfc"):
        from ncps.torch import CfC, LTC
        from ncps.wirings import AutoNCP
        wiring = AutoNCP(units=units, output_size=1, sparsity_level=0.5)
        cls = LTC if arch == "ltc" else CfC
        return cls(input_size=n_features, units=wiring, batch_first=True)
    if arch == "ctrnn":
        try:
            from models.ctrnn_model import _CTRNNNetwork
        except ImportError:
            from src.models.ctrnn_model import _CTRNNNetwork
        return _CTRNNNetwork(n_features=n_features, hidden_size=units, dt=1.0)
    if arch == "lstm":
        return nn.LSTM(input_size=n_features, hidden_size=units,
                       num_layers=1, batch_first=True)
    raise ValueError(f"build_bench_model: unknown arch '{arch}'")
