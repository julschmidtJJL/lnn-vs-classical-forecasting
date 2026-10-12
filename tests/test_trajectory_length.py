"""Tests for the pure trajectory-length core (src/analysis/trajectory_length.py)."""

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.analysis.trajectory_length import (
    arc_length,
    project_pca,
    trajectory_length,
    trajectory_length_normalized,
)


# ─── arc_length ──────────────────────────────────────────────────────────────

def test_arc_length_known_path():
    # (0,0) -> (3,0) -> (3,4): segments of length 3 and 4 -> total 7.
    path = np.array([[0.0, 0.0], [3.0, 0.0], [3.0, 4.0]])
    assert arc_length(path) == pytest.approx(7.0)


def test_arc_length_single_point_is_zero():
    assert arc_length(np.array([[1.0, 2.0, 3.0]])) == 0.0


def test_arc_length_constant_path_is_zero():
    path = np.tile([1.0, 1.0], (5, 1))  # no movement
    assert arc_length(path) == pytest.approx(0.0)


def test_arc_length_rejects_non_2d():
    with pytest.raises(ValueError):
        arc_length(np.array([1.0, 2.0, 3.0]))


# ─── project_pca ─────────────────────────────────────────────────────────────

def test_project_pca_var_explained_sums_to_one_for_full_rank():
    rng = np.random.default_rng(0)
    states = rng.standard_normal((50, 2))  # rank 2, k=2 -> captures everything
    _, var = project_pca(states, k=2)
    assert var.sum() == pytest.approx(1.0, abs=1e-9)
    assert (var >= 0).all()


def test_project_pca_line_puts_all_variance_on_pc1():
    # Points on a 1-D line embedded in 5-D: PC1 should capture ~all variance.
    t = np.linspace(-1, 1, 20).reshape(-1, 1)
    direction = np.array([[1.0, 2.0, -1.0, 0.5, 3.0]])
    states = t @ direction
    proj, var = project_pca(states, k=2)
    assert var[0] == pytest.approx(1.0, abs=1e-9)
    assert var[1] == pytest.approx(0.0, abs=1e-9)
    assert proj.shape == (20, 2)


def test_project_pca_pads_when_dim_below_k():
    states = np.linspace(0, 1, 10).reshape(-1, 1)  # d=1 < k=2
    proj, var = project_pca(states, k=2)
    assert proj.shape == (10, 2)
    assert var.shape == (2,)
    assert proj[:, 1] == pytest.approx(np.zeros(10))  # padded column
    assert var[1] == 0.0


def test_project_pca_constant_path_zero_variance():
    states = np.ones((8, 4))
    _, var = project_pca(states, k=2)
    assert var.sum() == pytest.approx(0.0)


# ─── trajectory_length (composition) ─────────────────────────────────────────

def test_trajectory_length_straight_line_equals_endpoint_distance():
    # A straight latent path: arc length in the 2-D projection equals the
    # distance between its endpoints (PC1 holds the whole line).
    t = np.linspace(0, 1, 11).reshape(-1, 1)
    direction = np.array([[3.0, 4.0, 0.0]])  # endpoint at (3,4,0): |.|=5
    states = t @ direction
    L, var = trajectory_length(states, k=2)
    assert L == pytest.approx(5.0, abs=1e-9)
    assert var[0] == pytest.approx(1.0, abs=1e-9)


def test_trajectory_length_is_projection_invariant_to_rotation():
    # Rotating the latent space must not change the measured length (PCA is
    # rotation-equivariant; arc length is rotation-invariant).
    rng = np.random.default_rng(1)
    states = np.cumsum(rng.standard_normal((30, 4)), axis=0)
    L1, _ = trajectory_length(states, k=2)
    # random rotation in 4-D
    q, _ = np.linalg.qr(rng.standard_normal((4, 4)))
    L2, _ = trajectory_length(states @ q, k=2)
    assert L1 == pytest.approx(L2, rel=1e-9)


# ─── trajectory_length_normalized (scale-invariant L_norm) ───────────────────

@pytest.mark.parametrize("c", [1e-3, 7.0, 1000.0, -5.0])
def test_trajectory_length_normalized_is_scale_invariant(c):
    # THE comparability property: rescaling the latent state by any constant
    # leaves L_norm unchanged, so the measure compares expressivity across
    # architectures on different latent scales rather than rewarding big norms.
    rng = np.random.default_rng(2)
    states = np.cumsum(rng.standard_normal((40, 5)), axis=0)
    L_ref, _ = trajectory_length_normalized(states, k=2)
    L_scaled, _ = trajectory_length_normalized(c * states, k=2)
    assert L_ref == pytest.approx(L_scaled, rel=1e-9)
    assert L_ref > 0.0  # a moving path has positive normalised length


def test_trajectory_length_normalized_constant_path_is_zero_not_nan():
    # Zero spatial extent -> no motion to normalise -> 0.0, never 0/0 = nan.
    L, _ = trajectory_length_normalized(np.ones((6, 4)), k=2)
    assert L == 0.0
    assert not math.isnan(L)


def test_trajectory_length_normalized_single_point_is_zero():
    L, var = trajectory_length_normalized(np.array([[1.0, 2.0, 3.0]]), k=2)
    assert L == 0.0
    assert var.shape == (2,)


def test_trajectory_length_normalized_differs_from_raw_under_rescale():
    # Sanity that L_norm actually removes scale where raw L does not: raw arc
    # length scales with c, L_norm does not.
    rng = np.random.default_rng(3)
    states = np.cumsum(rng.standard_normal((25, 3)), axis=0)
    L_raw_1, _ = trajectory_length(states, k=2)
    L_raw_10, _ = trajectory_length(10.0 * states, k=2)
    assert L_raw_10 == pytest.approx(10.0 * L_raw_1, rel=1e-9)  # raw IS scale-sensitive
    Ln_1, _ = trajectory_length_normalized(states, k=2)
    Ln_10, _ = trajectory_length_normalized(10.0 * states, k=2)
    assert Ln_1 == pytest.approx(Ln_10, rel=1e-9)              # normalised is NOT
