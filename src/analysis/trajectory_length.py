"""
Latent trajectory-length expressivity measure (Raghu et al. 2017, *On the
Expressive Power of Deep Neural Networks*; Hasani et al. 2021, LTC, Thm 4/5).

The expressivity of a (continuous-time) recurrent model can be probed by the
*arc length* its latent state traces as it processes an input sequence:

    L = sum_k || z_{k+1} - z_k ||_2

measured on the first 2 principal components of the latent path (per Hasani
Fig 2/3). Longer L = more expressive latent dynamics. The theoretical ordering
is L(LTC) >= L(CT-RNN) >= L(Neural-ODE).

This module holds the PURE, architecture-agnostic numeric core
(`arc_length`, `project_pca`, `trajectory_length`). Per-architecture latent
extraction and the bench runner live elsewhere (see run_expressivity.py); this
file has zero torch / model dependencies so it is trivially unit-testable, the
same split used by aggregate_seeds.py.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def arc_length(path: np.ndarray) -> float:
    """Arc length of a discrete path: sum of L2 norms of consecutive steps.

    ``path`` is shape (T, d) (T points in d dimensions). A single point (or
    empty path) has length 0.0 (no segments).
    """
    p = np.asarray(path, dtype=float)
    if p.ndim != 2:
        raise ValueError(f"path must be 2-D (T, d); got shape {p.shape}")
    if p.shape[0] < 2:
        return 0.0
    steps = np.diff(p, axis=0)                       # (T-1, d)
    return float(np.linalg.norm(steps, axis=1).sum())


def project_pca(states: np.ndarray, k: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """Project a latent path onto its first ``k`` principal components.

    ``states`` is shape (T, d). Returns ``(proj, var_explained)`` where ``proj``
    is (T, k) and ``var_explained`` is (k,), the fraction of total variance on
    each retained component. This is what makes latent paths of different
    dimensionality (e.g. a 16-unit NCP vs a 128-dim LSTM) comparable: both are
    reduced to the same k-dimensional space, and ``var_explained`` exposes how
    faithful that 2-D view is.

    Edge cases, made explicit so the measure never silently lies:
      - d < k: the missing components are zero-padded (both proj cols and
        var_explained), so the return shape is always (T, k) / (k,).
      - zero-variance path (constant, or T < 2): var_explained is all zeros.
    """
    X = np.asarray(states, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"states must be 2-D (T, d); got shape {X.shape}")
    T, d = X.shape
    Xc = X - X.mean(axis=0, keepdims=True)
    # SVD of the centered data: columns of Vt are principal directions, S^2
    # proportional to variance along each.
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    k_eff = min(k, Vt.shape[0])
    proj = np.zeros((T, k), dtype=float)
    if k_eff > 0:
        proj[:, :k_eff] = Xc @ Vt[:k_eff].T
    var_explained = np.zeros(k, dtype=float)
    total = float((S ** 2).sum())
    if total > 0 and k_eff > 0:
        var_explained[:k_eff] = (S[:k_eff] ** 2) / total
    return proj, var_explained


def trajectory_length(states: np.ndarray, k: int = 2) -> Tuple[float, np.ndarray]:
    """Convenience: project a latent path to ``k`` PCs, then measure arc length.

    Returns ``(L, var_explained)``. This is the headline expressivity number for
    one (model, input-sequence) pair; callers aggregate it over many sequences
    and seeds (mean +/- std).
    """
    proj, var_explained = project_pca(states, k=k)
    return arc_length(proj), var_explained


def trajectory_length_normalized(states: np.ndarray, k: int = 2) -> Tuple[float, np.ndarray]:
    """Scale-invariant trajectory length, for comparing expressivity ACROSS
    architectures whose latent states live on different arbitrary scales.

        L_norm = arc_length(proj) / ((T - 1) * RMS_radius(proj))

    where ``RMS_radius = sqrt(mean_t || proj_t - centroid ||^2)`` is the
    root-mean-square distance of the k-PC projected path from its centroid.
    Dividing the arc length by ``(T - 1) * RMS_radius`` makes the measure
    dimensionless and invariant to any global rescaling of the latent state
    (``states -> c * states`` leaves ``L_norm`` unchanged): both arc length and
    RMS radius scale linearly with ``|c|``, so the ratio cancels it.

    This is the cross-architecture comparability fix (thesis "open the black
    box" chapter). RAW ``trajectory_length`` is NOT comparable across model
    classes, because a model whose hidden state simply has a larger norm traces
    a longer path without being more expressive -- a scale artifact, not
    expressivity. ``L_norm`` removes that confound; keep raw ``L`` +
    ``var_explained`` for the appendix.

    Returns ``(L_norm, var_explained)``. A path with < 2 points or zero spatial
    extent (constant latent, ``RMS_radius == 0``) has ``L_norm`` 0.0 -- there is
    no motion to normalise, so we return 0.0 rather than 0/0 = nan.
    """
    proj, var_explained = project_pca(states, k=k)
    T = proj.shape[0]
    if T < 2:
        return 0.0, var_explained
    centroid = proj.mean(axis=0, keepdims=True)
    rms_radius = float(np.sqrt(((proj - centroid) ** 2).sum(axis=1).mean()))
    if rms_radius == 0.0:
        return 0.0, var_explained
    return arc_length(proj) / ((T - 1) * rms_radius), var_explained
