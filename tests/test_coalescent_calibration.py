"""
Tests for comp_alpha_m and comp_L_ij (allelic effect and pleiotropic loci for a target G_m).

Run from the repository root with:  python -m pytest -v
The model: G_ii = alpha_i^2 * mu * L_i * (2N - 1) / 2   (semidominant effect alpha),
           G_ij = alpha_i * alpha_j * mu * L_ij * (2N - 1) / 2.
"""
import numpy as np
import pytest

import hanoi

# Example inputs used in several tests.
G = np.array([[1.0, 0.3], [0.3, 2.0]])
L = np.array([100, 200])
MU = 1e-3
N = 1000


# ---------- comp_alpha_m ----------

def test_comp_alpha_m_gives_back_target_variance():
    # Put alpha into the forward formula. We must get the diagonal of G again.
    alpha = hanoi.comp_alpha_m(G, L, MU, N)
    G_back = alpha**2 * MU * L * (2 * N - 1) / 2
    assert np.allclose(G_back, np.diag(G))


def test_comp_alpha_m_known_value():
    # One trait: alpha = sqrt(2 * G / (mu * L * (2N - 1))). Here 2 / (1 * 1 * 3) -> sqrt(2/3).
    alpha = hanoi.comp_alpha_m([[1.0]], [1], 1.0, 2)
    assert alpha.shape == (1,)
    assert np.isclose(alpha[0], np.sqrt(2 / 3))


def test_comp_alpha_m_accepts_lists_and_arrays():
    # The docstring says array-like, so lists and arrays must give the same answer.
    from_lists = hanoi.comp_alpha_m(G.tolist(), L.tolist(), MU, N)
    from_arrays = hanoi.comp_alpha_m(G, L, MU, N)
    assert np.allclose(from_lists, from_arrays)


def test_comp_alpha_m_is_positive_and_ignores_covariance_sign():
    G_neg = np.array([[1.0, -0.3], [-0.3, 2.0]])
    alpha = hanoi.comp_alpha_m(G_neg, L, MU, N)
    assert np.all(alpha > 0)
    assert np.allclose(alpha, hanoi.comp_alpha_m(G, L, MU, N))


def test_comp_alpha_m_does_not_change_inputs():
    G_copy, L_copy = G.copy(), L.copy()
    hanoi.comp_alpha_m(G, L, MU, N)
    assert np.array_equal(G, G_copy) and np.array_equal(L, L_copy)


# ---------- comp_L_ij ----------

def test_comp_L_ij_known_values():
    # Worked by hand for the example inputs: diagonal = L, off-diagonal = 30.
    L_out = hanoi.comp_L_ij(G, L, MU, N)
    assert np.allclose(L_out, [[100, 30], [30, 200]])


def test_comp_L_ij_gives_back_target_covariance():
    # Forward formula: G_ij = alpha_i * alpha_j * mu * L_ij * (2N - 1) / 2.
    alpha = hanoi.comp_alpha_m(G, L, MU, N)
    L_out = hanoi.comp_L_ij(G, L, MU, N)
    G_back = np.outer(alpha, alpha) * MU * L_out * (2 * N - 1) / 2
    assert np.isclose(G_back[0, 1], G[0, 1])


def test_comp_L_ij_is_symmetric_and_diagonal_is_input():
    L_out = hanoi.comp_L_ij(G, L, MU, N)
    assert np.allclose(L_out, L_out.T)
    assert np.array_equal(np.diag(L_out), L)


def test_comp_L_ij_three_traits():
    # The old code only looped over 2 traits; check that 3 traits work.
    G3 = np.array([[1.0, 0.2, 0.1], [0.2, 2.0, 0.3], [0.1, 0.3, 1.5]])
    L3 = np.array([100, 200, 150])
    L_out = hanoi.comp_L_ij(G3, L3, MU, N)
    alpha = hanoi.comp_alpha_m(G3, L3, MU, N)
    G_back = np.outer(alpha, alpha) * MU * L_out * (2 * N - 1) / 2
    off_diag = ~np.eye(3, dtype=bool)
    assert np.allclose(G_back[off_diag], G3[off_diag])


def test_comp_L_ij_uses_size_of_covariance_only():
    # The sign of the covariance is dropped, so L_ij is never negative.
    G_neg = np.array([[1.0, -0.3], [-0.3, 2.0]])
    L_out = hanoi.comp_L_ij(G_neg, L, MU, N)
    assert np.all(L_out >= 0)
    assert np.allclose(L_out, hanoi.comp_L_ij(G, L, MU, N))


def test_comp_L_ij_one_trait():
    L_out = hanoi.comp_L_ij([[1.0]], [100], MU, N)
    assert L_out.shape == (1, 1)
    assert L_out[0, 0] == 100
