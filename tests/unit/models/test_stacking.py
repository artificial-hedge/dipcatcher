"""Bayesian stacking / pseudo-BMA(+): weight recovery, dominance, fail-closed.

All experiments are SYNTHETIC correctness checks with seeded rngs — never
market evidence (lab honesty contract). Scores are proper (log score, CRPS);
no Sharpe/P&L content.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import brentq
from scipy.stats import norm

from quant_fund.models.stacking import (
    gaussian_log_dens_matrix,
    mixture_log_density,
    pseudo_bma_weights,
    stacked_cdf,
    stacked_crps,
    stacked_log_score,
    stacked_predictive_samples,
    stacked_quantiles,
    stacking_weights,
)

Array = np.ndarray

LEVELS = np.linspace(0.005, 0.995, 199)


def _bimodal_case(seed: int, n: int) -> tuple[Array, Array, Array]:
    """SYNTHETIC regime mixture: y = ±3 + N(0, 0.7); two regime experts + one
    diffuse model. No single Gaussian can cover both regimes — the textbook
    setting where a stacked mixture must dominate every single forecast."""
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=n)
    y = 3.0 * signs + rng.normal(0.0, 0.7, n)
    mu = np.column_stack([np.full(n, 3.0), np.full(n, -3.0), np.zeros(n)])
    sigma = np.column_stack([np.full(n, 1.0), np.full(n, 1.0), np.full(n, 5.0)])
    return y, mu, sigma


# ---------------------------------------------------------------------------
# stacking weights: recovery, dominance, simplex, determinism
# ---------------------------------------------------------------------------


def test_stacking_recovers_true_generator() -> None:
    rng = np.random.default_rng(101)
    n = 1200
    y = rng.normal(0.5, 1.0, n)  # truth: N(0.5, 1) — model 0 below
    mu = np.column_stack([np.full(n, 0.5), np.zeros(n), np.full(n, 0.5)])
    sigma = np.column_stack([np.full(n, 1.0), np.full(n, 4.0), np.full(n, 0.4)])
    ld = gaussian_log_dens_matrix(y, mu, sigma)
    w = stacking_weights(ld)
    assert np.argmax(w) == 0
    assert w[0] > 0.9  # informative model dominates the simplex
    assert w.sum() == pytest.approx(1.0, abs=1e-9)
    assert np.all(w >= 0.0)


def test_stacking_dominates_best_single_and_equal_logscore() -> None:
    y, mu, sigma = _bimodal_case(202, 1000)
    ld = gaussian_log_dens_matrix(y, mu, sigma)
    w = stacking_weights(ld)
    ls_stack = stacked_log_score(ld, w)
    ls_best_single = max(stacked_log_score(ld, np.eye(3)[k]) for k in range(3))
    ls_equal = stacked_log_score(ld, np.full(3, 1.0 / 3.0))
    assert ls_stack >= ls_best_single - 1e-6  # vertices are feasible
    assert ls_stack > ls_equal + 0.05  # equal weight keeps 1/3 mass on the diffuse model
    assert w[0] == pytest.approx(w[1], abs=0.10)  # symmetric regime experts
    assert w[2] < 0.10  # the uninformative component is (near-)excluded


def test_stacking_end_to_end_pipeline_synthetic() -> None:
    """Lane demo: three Gaussian forecasters, one is the true generator."""
    rng = np.random.default_rng(1313)
    n = 1000
    y = rng.normal(0.0, 1.0, n)
    mu = np.column_stack([np.zeros(n), np.zeros(n), np.full(n, 0.8)])
    sigma = np.column_stack([np.ones(n), np.full(n, 2.0), np.ones(n)])
    ld = gaussian_log_dens_matrix(y, mu, sigma)
    w = stacking_weights(ld)
    assert np.argmax(w) == 0 and w[0] > 0.95
    ls = stacked_log_score(ld, w)
    assert ls >= max(stacked_log_score(ld, np.eye(3)[k]) for k in range(3)) - 1e-6
    assert ls > stacked_log_score(ld, np.full(3, 1.0 / 3.0))
    w_bb = pseudo_bma_weights(ld, bb=True, n_boot=128, seed=3)
    assert np.argmax(w_bb) == 0


def test_stacking_identical_models_uniform() -> None:
    rng = np.random.default_rng(606)
    n = 300
    y = rng.normal(0.0, 1.0, n)
    ld = gaussian_log_dens_matrix(y, np.zeros((n, 3)), np.ones((n, 3)))
    w = stacking_weights(ld)
    np.testing.assert_allclose(w, np.full(3, 1.0 / 3.0), atol=1e-6)


def test_stacking_single_model_weight_is_one() -> None:
    ld = np.array([[-1.0], [-2.0], [-1.5]])
    np.testing.assert_array_equal(stacking_weights(ld), np.ones(1))


def test_stacking_simplex_membership_random_cases() -> None:
    rng = np.random.default_rng(1212)
    for trial in range(5):
        ld = rng.normal(-2.0, 1.5, size=(120, 4))
        for w in (stacking_weights(ld), pseudo_bma_weights(ld, n_boot=32, seed=trial)):
            assert w.shape == (4,)
            assert np.all(w >= 0.0)
            assert w.sum() == pytest.approx(1.0, abs=1e-12)


def test_stacking_determinism_pinned() -> None:
    y, mu, sigma = _bimodal_case(505, 600)
    ld = gaussian_log_dens_matrix(y, mu, sigma)
    w1 = stacking_weights(ld, seed=11)
    w2 = stacking_weights(ld, seed=11)
    np.testing.assert_array_equal(w1, w2)


# ---------------------------------------------------------------------------
# pseudo-BMA / pseudo-BMA+
# ---------------------------------------------------------------------------


def test_pseudo_bma_plain_is_softmax_of_elpd() -> None:
    y, mu, sigma = _bimodal_case(303, 800)
    ld = gaussian_log_dens_matrix(y, mu, sigma)
    w = pseudo_bma_weights(ld, bb=False)
    elpd = ld.sum(axis=0)
    expected = np.exp(elpd - elpd.max())
    expected /= expected.sum()
    np.testing.assert_allclose(w, expected, atol=1e-12)
    assert w.sum() == pytest.approx(1.0)
    assert np.argmax(w) == int(np.argmax(elpd))


def test_pseudo_bma_plus_bb_regularizes() -> None:
    y, mu, sigma = _bimodal_case(404, 800)
    ld = gaussian_log_dens_matrix(y, mu, sigma)
    w_plain = pseudo_bma_weights(ld, bb=False)
    w_bb = pseudo_bma_weights(ld, bb=True, n_boot=256, seed=7)
    assert w_bb.sum() == pytest.approx(1.0, abs=1e-12)
    assert np.all(w_bb >= 0.0)
    assert w_bb.max() < w_plain.max()  # bootstrap tempers winner-take-all
    assert np.count_nonzero(w_bb > 1e-6) >= 2
    assert np.argmax(w_bb) == np.argmax(w_plain)  # regularization, not reversal


def test_pseudo_bma_bb_determinism_and_seed_sensitivity() -> None:
    y, mu, sigma = _bimodal_case(707, 500)
    ld = gaussian_log_dens_matrix(y, mu, sigma)
    b1 = pseudo_bma_weights(ld, n_boot=32, seed=1)
    b2 = pseudo_bma_weights(ld, n_boot=32, seed=1)
    b3 = pseudo_bma_weights(ld, n_boot=32, seed=2)
    np.testing.assert_array_equal(b1, b2)
    assert not np.allclose(b1, b3)


# ---------------------------------------------------------------------------
# stacked predictive: quantiles, CDF, samples
# ---------------------------------------------------------------------------


def test_stacked_quantiles_single_component_identity() -> None:
    q = (norm.ppf(LEVELS) * 1.5 + 0.3).reshape(1, -1)
    taus = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
    out = stacked_quantiles(q, np.ones(1), LEVELS, taus)
    np.testing.assert_allclose(out, norm.ppf(taus) * 1.5 + 0.3, atol=1e-2)


def test_stacked_quantiles_matches_analytic_mixture() -> None:
    # Uniform-in-z probability grid: uniform-in-probability knots are too
    # sparse in the tails, and the mixture median sits on a flat CDF
    # plateau where the inversion is ill-conditioned.
    levels = np.asarray(norm.cdf(np.linspace(-4.5, 4.5, 361)), dtype=float)
    z = np.linspace(-4.5, 4.5, 361)
    q = np.vstack([z - 3.0, z + 3.0])
    w = np.array([0.5, 0.5])
    taus = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
    out = stacked_quantiles(q, w, levels, taus)

    def mix_cdf(z: float) -> float:
        return 0.5 * norm.cdf(z + 3.0) + 0.5 * norm.cdf(z - 3.0)

    expected = np.array([brentq(lambda z, t=t: mix_cdf(z) - t, -12.0, 12.0) for t in taus])
    np.testing.assert_allclose(out, expected, atol=0.05)
    assert out[2] == pytest.approx(0.0, abs=0.05)  # symmetric median
    assert np.all(np.diff(out) > 0.0)


def test_stacked_quantiles_batched_shape() -> None:
    rng = np.random.default_rng(2323)
    n, q = 7, LEVELS.size
    quantiles = np.empty((n, 2, q))
    quantiles[:, 0, :] = norm.ppf(LEVELS) - 1.0
    quantiles[:, 1, :] = norm.ppf(LEVELS) + 1.0
    quantiles += rng.normal(0.0, 0.01, (n, 2, q))
    quantiles = np.sort(quantiles, axis=-1)  # keep grids non-crossing
    taus = np.array([0.25, 0.5, 0.75])
    out = stacked_quantiles(quantiles, np.array([0.5, 0.5]), LEVELS, taus)
    assert out.shape == (n, 3)
    assert np.all(np.diff(out, axis=1) >= 0.0)


def test_stacked_cdf_monotone_and_symmetric() -> None:
    q = np.vstack([norm.ppf(LEVELS) - 2.0, norm.ppf(LEVELS) + 2.0])
    w = np.array([0.5, 0.5])
    z = np.linspace(-8.0, 8.0, 321)
    f = stacked_cdf(z, q, w, LEVELS)
    assert np.all(np.diff(f) >= 0.0)
    assert f[0] >= 0.0 and f[-1] <= 1.0
    assert f[160] == pytest.approx(0.5, abs=0.02)  # z = 0 by symmetry
    np.testing.assert_allclose(
        stacked_cdf(np.array([0.0]), q[:1], np.ones(1), LEVELS),
        [norm.cdf(2.0)],
        atol=0.02,
    )


def test_stacked_predictive_samples_mixture_stats() -> None:
    rng = np.random.default_rng(808)
    s = 4000
    samples = np.column_stack([rng.normal(-3.0, 1.0, s), rng.normal(3.0, 1.0, s)])
    w = np.array([0.5, 0.5])
    d1 = stacked_predictive_samples(samples, w, n_draws=6000, seed=5)
    d2 = stacked_predictive_samples(samples, w, n_draws=6000, seed=5)
    np.testing.assert_array_equal(d1, d2)  # determinism pinned
    assert d1.mean() == pytest.approx(0.0, abs=0.15)
    assert d1.std() == pytest.approx(np.sqrt(10.0), abs=0.25)
    assert float(np.mean(d1 > 0.0)) == pytest.approx(0.5, abs=0.03)


# ---------------------------------------------------------------------------
# proper-score helpers
# ---------------------------------------------------------------------------


def test_mixture_log_density_matches_manual() -> None:
    ld = np.array([[np.log(0.5), np.log(0.1)], [np.log(1.0), np.log(2.0)]])
    w = np.array([0.25, 0.75])
    got = mixture_log_density(ld, w)
    expected = np.log(np.array([[0.5, 0.1], [1.0, 2.0]]) @ w)
    np.testing.assert_allclose(got, expected, rtol=1e-12)
    # Zero-weight components are ignored (log 0 handled inside logsumexp).
    np.testing.assert_allclose(mixture_log_density(ld, np.array([1.0, 0.0])), ld[:, 0])


def test_gaussian_log_dens_matrix_matches_norm_logpdf() -> None:
    rng = np.random.default_rng(111)
    n = 50
    y = rng.normal(size=n)
    mu = rng.normal(size=(n, 2))
    sigma = np.exp(rng.normal(size=(n, 2)))
    ld = gaussian_log_dens_matrix(y, mu, sigma)
    np.testing.assert_allclose(ld, norm.logpdf(y[:, None], mu, sigma), rtol=1e-10)


def test_stacked_crps_beats_every_single_component() -> None:
    rng = np.random.default_rng(909)
    n = 400
    signs = rng.choice(np.array([-1.0, 1.0]), size=n)
    y = 3.0 * signs + rng.normal(0.0, 0.7, n)
    levels = np.linspace(0.01, 0.99, 99)
    z = norm.ppf(levels)
    q = np.empty((n, 3, levels.size))
    q[:, 0, :] = 3.0 + z
    q[:, 1, :] = -3.0 + z
    q[:, 2, :] = 8.0 * z  # deliberately too diffuse — worst single under CRPS too
    w_stack = np.array([0.5, 0.5, 0.0])
    crps_stack = stacked_crps(y, q, w_stack, levels)
    singles = [stacked_crps(y, q, np.eye(3)[k], levels) for k in range(3)]
    assert crps_stack < min(singles)


# ---------------------------------------------------------------------------
# fail-closed edges
# ---------------------------------------------------------------------------


def test_fail_closed_log_dens_inputs() -> None:
    good = np.array([[-1.0, -2.0], [-1.5, -1.2], [-0.9, -2.5]])
    with pytest.raises(ValueError):
        stacking_weights(np.empty((0, 2)))  # empty
    with pytest.raises(ValueError):
        stacking_weights(good.reshape(-1))  # 1d
    bad_nan = good.copy()
    bad_nan[0, 0] = np.nan
    with pytest.raises(ValueError):
        stacking_weights(bad_nan)
    with pytest.raises(ValueError):
        pseudo_bma_weights(bad_nan)
    bad_inf = good.copy()
    bad_inf[1, 1] = -np.inf
    with pytest.raises(ValueError):
        stacking_weights(bad_inf)
    with pytest.raises(ValueError):
        stacking_weights(good, n_starts=0)
    with pytest.raises(ValueError):
        pseudo_bma_weights(good, n_boot=0)


def test_fail_closed_weights_validation() -> None:
    good = np.array([[-1.0, -2.0], [-1.5, -1.2], [-0.9, -2.5]])
    bad_weights = [
        np.array([-0.5, 1.5]),  # negative
        np.zeros(2),  # all-zero (non-positive total mass)
        np.array([0.5, 0.5, 0.0]),  # length mismatch
        np.array([np.nan, 1.0]),  # non-finite
    ]
    for w in bad_weights:
        with pytest.raises(ValueError):
            mixture_log_density(good, w)
        with pytest.raises(ValueError):
            stacked_log_score(good, w)


def test_fail_closed_stacked_quantiles_inputs() -> None:
    levels = np.linspace(0.05, 0.95, 19)
    q = np.tile(norm.ppf(levels), (2, 1))
    w = np.array([0.5, 0.5])
    with pytest.raises(ValueError):
        stacked_quantiles(q, w, levels, np.array([0.01]))  # below levels[0]
    with pytest.raises(ValueError):
        stacked_quantiles(q, w, levels, np.array([0.99]))  # above levels[-1]
    with pytest.raises(ValueError):
        stacked_quantiles(q, w, np.array([0.5, 0.25]), np.array([0.4]))  # not increasing
    with pytest.raises(ValueError):
        stacked_quantiles(q, w, np.array([0.0, 0.5, 1.0]), np.array([0.5]))  # closed ends
    q_cross = q.copy()
    q_cross[0, 5] = q_cross[0, 4] - 1.0
    with pytest.raises(ValueError):
        stacked_quantiles(q_cross, w, levels, np.array([0.5]))  # crossing grid
    with pytest.raises(ValueError):
        stacked_quantiles(np.zeros((2, levels.size)), w, levels, np.array([0.5]))  # degenerate
    with pytest.raises(ValueError):
        stacked_quantiles(q, w, levels, np.array([0.5]), n_grid=8)  # grid too coarse
    with pytest.raises(ValueError):
        stacked_quantiles(q[0], w, levels, np.array([0.5]))  # 1d quantiles


def test_fail_closed_samples_and_crps_and_gaussian() -> None:
    rng = np.random.default_rng(1414)
    samples = rng.normal(size=(10, 2))
    w2 = np.array([0.5, 0.5])
    with pytest.raises(ValueError):
        stacked_predictive_samples(samples, w2, n_draws=0)
    with pytest.raises(ValueError):
        stacked_predictive_samples(samples, np.array([0.5, 0.5, 0.0]), n_draws=5)
    with pytest.raises(ValueError):
        stacked_predictive_samples(np.empty((0, 2)), w2, n_draws=5)
    bad_samples = samples.copy()
    bad_samples[0, 0] = np.inf
    with pytest.raises(ValueError):
        stacked_predictive_samples(bad_samples, w2, n_draws=5)

    levels = np.linspace(0.05, 0.95, 19)
    q = np.tile(norm.ppf(levels), (6, 2, 1))
    y = rng.normal(size=6)
    with pytest.raises(ValueError):
        stacked_crps(y[:5], q, w2, levels)  # row mismatch
    with pytest.raises(ValueError):
        stacked_crps(y, q[0], w2, levels)  # 2d not allowed
    with pytest.raises(ValueError):
        stacked_crps(np.empty(0), np.empty((0, 2, levels.size)), w2, levels)  # empty

    with pytest.raises(ValueError):
        gaussian_log_dens_matrix(y, np.zeros((6, 2)), np.zeros((6, 2)))  # sigma <= 0
    with pytest.raises(ValueError):
        gaussian_log_dens_matrix(y, np.zeros((5, 2)), np.ones((5, 2)))  # shape mismatch
    with pytest.raises(ValueError):
        gaussian_log_dens_matrix(np.empty(0), np.empty((0, 2)), np.ones((0, 2)))  # empty
