"""
Tests for the log-additive noise model: comp_breed_val, comp_varcov and sim_pheno.

Run from the repository root with:  python -m pytest -v
A test is a function whose name starts with `test_`. It passes if it finishes
without raising, and fails if an `assert` is false or an exception escapes.
"""
import numpy as np
import pytest
from scipy.spatial.distance import squareform

import hanoi


def make_breed_val(mean, ln_var, artanh_cor):
    """Shortcut to build a BreedVal from nested lists."""
    return hanoi.BreedVal(
        mean=np.array(mean, dtype=float),
        ln_var=np.array(ln_var, dtype=float),
        artanh_cor=np.array(artanh_cor, dtype=float).reshape(-1, len(mean[0])),
    )


# A fixture is a function pytest calls to build an input; any test that lists
# its name as an argument receives the returned value.
@pytest.fixture
def bv_zero_3traits():
    """Breeding values of exactly zero for 3 traits and 4 individuals (= baseline genotype)."""
    return hanoi.BreedVal(
        mean=np.zeros((3, 4)), ln_var=np.zeros((3, 4)), artanh_cor=np.zeros((3, 4))
    )


# ---------- comp_breed_val ----------

def test_comp_breed_val_is_half_effect_times_genotype():
    # 1 trait, 2 loci, 2 individuals. Breeding value = 1/2 * effect @ genotype.
    me = hanoi.MutEffect(
        mean=np.array([[1.0, 2.0]]),
        ln_var=np.array([[0.5, -0.5]]),
        artanh_cor=np.zeros((0, 2)),
    )
    geno = np.array([[2, 0], [2, 2]])  # (L, N)
    bv = hanoi.comp_breed_val(me, geno)
    # individual 0: 1/2*(1*2 + 2*2) = 3 ; individual 1: 1/2*(1*0 + 2*2) = 2
    assert bv.mean.shape == (1, 2)
    # approx compares floats with a small tolerance instead of exact equality
    assert bv.mean[0] == pytest.approx([3.0, 2.0])
    assert bv.ln_var[0] == pytest.approx([0.0, -0.5])


# ---------- comp_varcov: values ----------

def test_comp_varcov_baseline_when_breeding_values_are_zero(bv_zero_3traits):
    v_0 = [1.0, 2.0, 3.0]
    r_0 = [0.1, 0.2, 0.3]  # pairs (0,1), (0,2), (1,2)
    vc = hanoi.comp_varcov(bv_zero_3traits, v_0, r_0)
    assert vc.shape == (4, 3, 3)
    for k in range(4):
        # diagonal = baseline variances
        assert np.diag(vc[k]) == pytest.approx(v_0)
        # correlations recovered from the covariance matrix = baseline correlations
        sd = np.sqrt(np.diag(vc[k]))
        cor = vc[k] / np.outer(sd, sd)
        assert squareform(cor, checks=False) == pytest.approx(r_0)


def test_comp_varcov_applies_log_and_fisher_scales():
    # 2 traits, 1 individual
    bv = make_breed_val(mean=[[0], [0]], ln_var=[[np.log(4.0)], [np.log(0.25)]],
                        artanh_cor=[[np.arctanh(0.5)]])
    vc = hanoi.comp_varcov(bv, v_0=[1.0, 1.0], r_0=[0.0])
    # variances are multiplied by exp(ln_var): 4 and 0.25
    assert vc[0, 0, 0] == pytest.approx(4.0)
    assert vc[0, 1, 1] == pytest.approx(0.25)
    # r = tanh(artanh(0) + artanh(0.5)) = 0.5 ; cov = r * sqrt(4 * 0.25) = 0.5
    assert vc[0, 0, 1] == pytest.approx(0.5)


def test_comp_varcov_is_symmetric(bv_zero_3traits):
    vc = hanoi.comp_varcov(bv_zero_3traits, [1, 1, 1], [0.1, 0.2, 0.3])
    assert np.allclose(vc, vc.transpose(0, 2, 1))


def test_comp_varcov_variance_stays_positive_for_extreme_breeding_values():
    # The point of the log scale: no breeding value can make variance <= 0,
    # and no breeding value can push |r| to 1 or beyond.
    bv = make_breed_val(mean=[[0, 0], [0, 0]], ln_var=[[-50, 50], [50, -50]],
                        artanh_cor=[[-20, 20]])
    vc = hanoi.comp_varcov(bv, [1.0, 1.0], [0.0])
    assert np.all(np.diagonal(vc, axis1=1, axis2=2) > 0)
    assert np.all(np.isfinite(vc))


def test_comp_varcov_one_trait_needs_no_r_0():
    bv = make_breed_val(mean=[[0, 0]], ln_var=[[0.0, np.log(2.0)]], artanh_cor=[[]])
    vc = hanoi.comp_varcov(bv, v_0=[3.0])
    assert vc.shape == (2, 1, 1)
    assert vc[:, 0, 0] == pytest.approx([3.0, 6.0])


# ---------- comp_varcov: invalid input ----------
# parametrize runs the same test once per tuple, reporting each separately.

@pytest.mark.parametrize("v_0, r_0, message", [
    ([1.0, 0.0, 1.0], [0.1, 0.1, 0.1], "positive"),         # zero variance
    ([1.0, -1.0, 1.0], [0.1, 0.1, 0.1], "positive"),        # negative variance
    ([1.0, 1.0], [0.1, 0.1, 0.1], "v_0 should be a 1D"),    # wrong length
    ([1.0, 1.0, 1.0], None, "r_0 is required"),             # missing r_0
    ([1.0, 1.0, 1.0], [0.1, 0.1], "choose"),                # wrong length
    ([1.0, 1.0, 1.0], [0.1, 1.0, 0.1], "strictly"),         # |r| = 1
])
def test_comp_varcov_rejects_bad_baselines(bv_zero_3traits, v_0, r_0, message):
    # pytest.raises passes only if the block raises the given error;
    # `match` additionally checks the error message (regular expression).
    with pytest.raises(ValueError, match=message):
        hanoi.comp_varcov(bv_zero_3traits, v_0, r_0)


def test_comp_varcov_rejects_matrix_that_is_not_positive_definite(bv_zero_3traits):
    # Every pairwise correlation is valid, but r12 = r13 = 0.9 and r23 = -0.9
    # cannot hold together in one 3x3 correlation matrix.
    with pytest.raises(ValueError, match="positive definite"):
        hanoi.comp_varcov(bv_zero_3traits, [1, 1, 1], [0.9, 0.9, -0.9])


# ---------- sim_pheno ----------

@pytest.mark.parametrize("n", [1, 2, 3])
def test_sim_pheno_returns_N_by_n(n):
    N = 7
    bv = hanoi.BreedVal(mean=np.zeros((n, N)), ln_var=np.zeros((n, N)),
                        artanh_cor=np.zeros((n * (n - 1) // 2, N)))
    r_0 = np.full(n * (n - 1) // 2, 0.2) if n > 1 else None
    z = hanoi.sim_pheno(bv, mean_0=np.zeros(n), v_0=np.ones(n), r_0=r_0)
    assert z.shape == (N, n)


def test_sim_pheno_same_seed_gives_same_result():
    bv = make_breed_val(mean=[[0, 1, 2]], ln_var=[[0, 0, 0]], artanh_cor=[[]])
    np.random.seed(5)
    a = hanoi.sim_pheno(bv, np.zeros(1), np.ones(1))
    np.random.seed(5)
    b = hanoi.sim_pheno(bv, np.zeros(1), np.ones(1))
    assert np.array_equal(a, b)


def test_sim_pheno_recovers_mean_and_variance_statistically():
    # Statistical test: with many individuals the sample mean/variance should be
    # close to the model values. Fixed seed keeps the test deterministic.
    N = 20000
    bv = hanoi.BreedVal(mean=np.full((1, N), 2.0), ln_var=np.full((1, N), np.log(4.0)),
                        artanh_cor=np.zeros((0, N)))
    np.random.seed(1)
    z = hanoi.sim_pheno(bv, mean_0=np.array([1.0]), v_0=np.array([1.0]))
    assert z.mean() == pytest.approx(3.0, abs=0.1)   # mean_0 + breeding value
    assert z.var() == pytest.approx(4.0, rel=0.1)    # v_0 * exp(ln_var)
