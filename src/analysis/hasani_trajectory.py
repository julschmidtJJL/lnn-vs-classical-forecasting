"""
Faithful Python port of Hasani et al.'s LTC-paper trajectory-length experiment
(``trajectory_length_analysis/`` in github.com/raminmh/liquid_time_constant_networks,
vendored at vendor/ltc_reference/). Measures the expressivity of three
continuous-time models by the arc length their latent state traces in 2-PC space
as a controlled input trajectory is fed in (Raghu et al. 2017; Hasani LTC Table 2).

The three models share weights/init and differ ONLY in the state dynamics
(``N``=hidden width, ``m``=input dim, activation = hard-tanh, Eq. refs are to the
LTC paper / the vendored .m files):

    Neural-ODE   dy/dt = f(W_in.x(t) + b_in)                       (node_def.m)
    CT-RNN       dy/dt = -y/tau + f(W_in.x(t) + b_in)              (ctrnn_def.m)
    LTC          dy/dt = -y*(1/tau + |f_in| + |f_rec|)
                         + c_in*f_in + c_rec*f_rec                 (ltc_def.m)

where f_in = f(W_in.x(t) + b_in), f_rec = f(W_rec.y + b_rec), and c_in/c_rec are
the per-neuron column sums of the reversal-potential matrices E_in/E_rec (exactly
``sum(reshape(E_l(1,:,i),k,1)*net_out(i))`` in the MATLAB). Init is the paper's:
weights ~ N(0, sigma), biases ~ N(0, sigma_b), tau = |N(0, sigma_b)|.

VALIDATION (matches the paper's qualitative result): at k=100 over 20 seeds the
LTC trajectory is ~14x longer than CT-RNN / Neural-ODE -- reproducing the LTC
expressivity dominance that the thesis's "superior expressivity" claim rests on.
Caveat: the minor Neural-ODE vs CT-RNN ordering comes out reversed vs Table 2
(we get CT-RNN < Neural-ODE), attributable to ode45-vs-RK45 stiffness handling;
the dominant LTC >> {CT-RNN, N-ODE} result is robust.

The headline novelty for the thesis is feeding a FINANCIAL input trajectory in
place of the synthetic circle: see ``circular_input`` vs passing your own
``inputs`` array of shape (m, steps) to ``trajectory_lengths``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
from scipy.integrate import solve_ivp
from sklearn.decomposition import PCA


def hard_tanh(x: np.ndarray) -> np.ndarray:
    """Htanh.m: clip to [-1, 1]."""
    return np.clip(x, -1.0, 1.0)


def arc_length_2pc(trajectory: np.ndarray) -> float:
    """Linear (chordal) arc length of a latent path in its first 2 principal
    components -- arclength.m default method on ``pca(...)`` scores.

    ``trajectory`` is (n_steps, hidden). Returns 0.0 for < 2 steps.
    """
    if trajectory.shape[0] < 2:
        return 0.0
    if not np.all(np.isfinite(trajectory)):
        # A divergent/stiff integration (e.g. an over-driven random-init model)
        # yields non-finite states; report NaN so callers can exclude the seed
        # rather than crashing PCA. Guarded again in the runner's finite mask.
        return float("nan")
    proj = PCA(n_components=2).fit_transform(trajectory)
    return float(np.sqrt((np.diff(proj, axis=0) ** 2).sum(axis=1)).sum())


def arc_length_kd(trajectory: np.ndarray) -> float:
    """Chordal arc length of a latent path in its FULL k-D state space.

    Same chordal sum as :func:`arc_length_2pc` but without the PCA projection,
    so motion in minor components still counts. Use as a companion metric: a
    much larger k-D length than 2-PC length means the trajectory wiggles in
    directions the dominant-variance shadow discards. Note the magnitude grows
    with k, so k-D lengths are NOT comparable across widths (the 2-PC metric is
    the cross-width / cross-architecture contrast). ``trajectory`` is
    (n_steps, hidden); returns 0.0 for < 2 steps, NaN for non-finite paths.
    """
    if trajectory.shape[0] < 2:
        return 0.0
    if not np.all(np.isfinite(trajectory)):
        return float("nan")
    return float(np.sqrt((np.diff(trajectory, axis=0) ** 2).sum(axis=1)).sum())


def circular_input(steps: int = 100) -> np.ndarray:
    """Hasani's controlled input: a unit circle [sin t; cos t] over [0, 2*pi].
    Returns shape (2, steps)."""
    gt = np.linspace(0.0, 2.0 * np.pi, steps)
    return np.vstack([np.sin(gt), np.cos(gt)])


@dataclass
class _Weights:
    W_in: np.ndarray   # (m, k)
    b_in: np.ndarray   # (k,)
    W_rec: np.ndarray  # (k, k)
    b_rec: np.ndarray  # (k,)
    c_in: np.ndarray   # (k,) column sums of E_in
    c_rec: np.ndarray  # (k,) column sums of E_rec
    tau: np.ndarray    # (k,)


def build_weights(seed: int, k: int, m: int,
                  sigma: float = 2.0, sigma_b: float = 1.0) -> _Weights:
    """Paper init (main.m): weights ~ N(0, sigma), biases ~ N(0, sigma_b),
    tau = |N(0, sigma_b)|. Deterministic given ``seed``. ``sigma`` is the
    standard deviation (MATLAB ``makedist('normal','sigma',weight_dist/k)`` with
    weight_dist = k*sigma collapses to std = sigma)."""
    rng = np.random.default_rng(seed)
    E_in = rng.normal(0.0, sigma, (k, k))
    E_rec = rng.normal(0.0, sigma, (k, k))
    return _Weights(
        W_in=rng.normal(0.0, sigma, (m, k)),
        b_in=rng.normal(0.0, sigma_b, k),
        W_rec=rng.normal(0.0, sigma, (k, k)),
        b_rec=rng.normal(0.0, sigma_b, k),
        c_in=E_in.sum(axis=0),
        c_rec=E_rec.sum(axis=0),
        tau=np.abs(rng.normal(0.0, sigma_b, k)),
    )


def integrate_models(inputs: np.ndarray, seed: int = 1, k: int = 100,
                     sigma: float = 2.0, sigma_b: float = 1.0,
                     rtol: float = 1e-2, atol: float = 1e-4) -> Dict[str, np.ndarray]:
    """Integrate all three models over ``inputs`` (shape (m, steps)) and return
    ``{'node','ctrnn','ltc'}`` -> the latent trajectory, each a (steps, k) array
    sampled on the same uniform input grid.

    Models are integrated with adaptive RK45 over [0, t_max] and sampled on the
    same uniform input grid (t_eval) so trajectories are compared at equal
    resolution. ``inputs`` may be the synthetic circle or a financial trajectory.
    Shared by ``trajectory_lengths`` (arc length) and figure code (raw paths).
    """
    m, steps = inputs.shape
    w = build_weights(seed, k, m, sigma=sigma, sigma_b=sigma_b)
    t_max = 2.0 * np.pi
    gt = np.linspace(0.0, t_max, steps)

    def x_of_t(t):
        return np.array([np.interp(t, gt, inputs[i]) for i in range(m)])

    def f_in(t):
        return hard_tanh(w.W_in.T @ x_of_t(t) + w.b_in)

    def f_rec(y):
        return hard_tanh(w.W_rec.T @ y + w.b_rec)

    rhs = {
        "node": lambda t, y: f_in(t),
        "ctrnn": lambda t, y: -y / w.tau + f_in(t),
        "ltc": lambda t, y: (-y * (1.0 / w.tau + np.abs(f_in(t)) + np.abs(f_rec(y)))
                             + w.c_in * f_in(t) + w.c_rec * f_rec(y)),
    }
    y0 = np.zeros(k)
    out: Dict[str, np.ndarray] = {}
    for name, fn in rhs.items():
        sol = solve_ivp(fn, (0.0, t_max), y0, method="RK45",
                        rtol=rtol, atol=atol, t_eval=gt)
        y = np.asarray(sol.y, dtype=float)
        # A solver that aborts early (stiff/divergent RHS) returns fewer columns
        # than t_eval; pad with NaN so every path is (steps, k) and the finite
        # guard downstream excludes it cleanly.
        if y.ndim != 2 or y.shape != (k, steps):
            padded = np.full((k, steps), np.nan)
            if y.ndim == 2 and y.shape[0] == k:
                padded[:, : y.shape[1]] = y[:, :steps]
            y = padded
        out[name] = y.T
    return out


def trajectory_lengths(inputs: np.ndarray, seed: int = 1, k: int = 100,
                       sigma: float = 2.0, sigma_b: float = 1.0,
                       rtol: float = 1e-2, atol: float = 1e-4) -> Dict[str, float]:
    """Integrate all three models over ``inputs`` (shape (m, steps)) and return
    ``{'node','ctrnn','ltc'}`` -> latent trajectory arc length in 2-PC space.

    Thin wrapper over ``integrate_models`` that reduces each latent path to its
    2-PC arc length. ``inputs`` may be the synthetic circle or a financial
    trajectory.
    """
    paths = integrate_models(inputs, seed=seed, k=k, sigma=sigma, sigma_b=sigma_b,
                             rtol=rtol, atol=atol)
    return {name: arc_length_2pc(path) for name, path in paths.items()}
