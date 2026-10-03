"""Canon tests: Fernholz stochastic-portfolio-theory analytics."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.metrics.fernholz_spt import (
    arbitrage_diagnostics,
    bench_fernholz_spt,
    diverse,
    diversity_arbitrage_horizon,
    diversity_drift_rate,
    diversity_grad_log_g,
    diversity_hess_log_g,
    diversity_index,
    diversity_log_g,
    diversity_master_equation,
    diversity_weights,
    entropy_grad_log_g,
    entropy_hess_log_g,
    entropy_master_equation,
    entropy_weights,
    excess_growth_rate,
    fg_drift_rate,
    fg_master_equation,
    fg_weights,
    first_passage_time,
    growth_optimal_weights,
    local_time_at_zero,
    log_cap_covariation,
    log_weight_covariation,
    market_entropy,
    market_weights,
    rank_gap_local_times,
    rank_gaps,
    rank_occupation_matrix,
    rank_permutations,
    ranked_weights,
    reflected_path,
    relative_log_returns,
    simulate_gbm_market,
)

DT = 1.0 / 1008.0


def _market(n_assets: int = 12, n_steps: int = 600, seed: int = 0) -> np.ndarray:
    return simulate_gbm_market(n_assets=n_assets, n_steps=n_steps, dt=DT, seed=seed)


# --- market weights / entropy / diversity -----------------------------------


def test_market_weights_rows_sum_to_one() -> None:
    mu = market_weights(_market())
    assert mu.shape[0] == 601
    np.testing.assert_allclose(mu.sum(axis=1), 1.0, atol=1e-12)


def test_market_weights_single_row() -> None:
    mu = market_weights(np.array([1.0, 3.0, 4.0]))
    np.testing.assert_allclose(mu, [[0.125, 0.375, 0.5]])


def test_market_weights_rejects_bad_caps() -> None:
    with pytest.raises(ValueError):
        market_weights(np.array([[1.0, -2.0]]))
    with pytest.raises(ValueError):
        market_weights(np.array([[1.0, 2.0], [np.nan, 3.0]]))
    with pytest.raises(ValueError):
        market_weights(np.array([[1.0, 0.0]]))


def test_entropy_uniform_equals_log_n() -> None:
    n = 9
    mu = np.full((1, n), 1.0 / n)
    assert market_entropy(mu)[0] == pytest.approx(np.log(n), abs=1e-12)


def test_entropy_vertex_limit_is_small() -> None:
    mu = np.array([[0.999999, 1e-6]])
    assert 0.0 < market_entropy(mu)[0] < 2e-5


def test_entropy_rejects_off_simplex() -> None:
    with pytest.raises(ValueError):
        market_entropy(np.array([[0.7, 0.7]]))
    with pytest.raises(ValueError):
        market_entropy(np.array([[1.0, 0.0]]))


def test_diversity_index_bounds() -> None:
    mu = market_weights(_market())
    d = diversity_index(mu, 0.5)
    n = mu.shape[1]
    assert np.all(d >= 1.0) and np.all(d <= n ** (0.5 / 0.5) + 1e-9)


def test_diversity_index_known_values() -> None:
    # p -> extremes: uniform gives n^{(1-p)/p}; a near-vertex gives ~1.
    uniform = np.full((1, 4), 0.25)
    assert diversity_index(uniform, 0.5)[0] == pytest.approx(4.0)
    concentrated = np.array([[0.97, 0.01, 0.01, 0.01]])
    # (sqrt(.97) + 3·sqrt(.01))² = 1.285² ≈ 1.65 — p < 1 amplifies tails.
    assert diversity_index(concentrated, 0.5)[0] == pytest.approx(1.6509, abs=1e-3)
    assert diversity_index(concentrated, 0.5)[0] < diversity_index(uniform, 0.5)[0]


def test_diverse_condition() -> None:
    assert diverse(np.array([[0.5, 0.3, 0.2]]), 0.4)
    assert not diverse(np.array([[0.7, 0.3]]), 0.4)
    with pytest.raises(ValueError):
        diverse(np.array([[0.5, 0.5]]), 1.5)


# --- rank processes ----------------------------------------------------------


def test_ranked_weights_sorted_and_sum_one() -> None:
    ranked = ranked_weights(market_weights(_market()))
    assert np.all(np.diff(ranked, axis=1) <= 1e-12)
    np.testing.assert_allclose(ranked.sum(axis=1), 1.0, atol=1e-12)


def test_rank_gaps_nonnegative_and_shape() -> None:
    mu = market_weights(_market(n_assets=7))
    gaps = rank_gaps(mu)
    assert gaps.shape == (601, 6)
    assert np.all(gaps >= 0.0)


def test_rank_permutations_consistent_with_sorted() -> None:
    mu = market_weights(_market(n_assets=6, n_steps=50))
    order = rank_permutations(mu)
    ranked = ranked_weights(mu)
    for t in range(mu.shape[0]):
        np.testing.assert_allclose(ranked[t], mu[t][order[t]], atol=1e-12)


def test_rank_occupation_matrix_rows_sum_to_one() -> None:
    mu = market_weights(_market(n_assets=8, n_steps=200))
    occ = rank_occupation_matrix(mu)
    np.testing.assert_allclose(occ.sum(axis=1), 1.0, atol=1e-12)
    np.testing.assert_allclose(occ.sum(axis=0), 1.0, atol=1e-12)
    # The stock holding rank 0 at the last date has nonzero top-rank time.
    leader = rank_permutations(mu)[-1, 0]
    assert occ[leader, 0] > 0.0


# --- FG weights --------------------------------------------------------------


def test_fg_weights_simplex_and_match_closed_form() -> None:
    mu = market_weights(_market())
    pi = fg_weights(mu, diversity_grad_log_g(mu, 0.5))
    np.testing.assert_allclose(pi.sum(axis=1), 1.0, atol=1e-10)
    assert np.all(pi >= 0.0)
    np.testing.assert_allclose(pi, diversity_weights(mu, 0.5), atol=1e-10)


def test_entropy_weights_match_fg_gradient() -> None:
    mu = market_weights(_market())
    np.testing.assert_allclose(
        entropy_weights(mu), fg_weights(mu, entropy_grad_log_g(mu)), atol=1e-10
    )


def test_entropy_weights_closed_form() -> None:
    mu = np.array([[0.5, 0.25, 0.25]])
    s = -float(np.sum(mu * np.log(mu)))
    expected = -mu * np.log(mu) / s
    np.testing.assert_allclose(entropy_weights(mu), expected, atol=1e-12)


def test_fg_weights_fail_closed_on_invalid_generator() -> None:
    mu = np.array([[0.5, 0.5]])
    # A gradient with huge negative slope is not a valid portfolio generator.
    bad_grad = np.array([[-1000.0, 0.0]])
    with pytest.raises(ValueError, match="not a valid portfolio generator"):
        fg_weights(mu, bad_grad)
    with pytest.raises(ValueError):
        fg_weights(mu, np.array([[np.nan, 0.0]]))


def _raw_log_s(x: np.ndarray) -> float:
    """log of the entropy generator evaluated off the simplex (ambient)."""
    return float(np.log(-np.sum(x * np.log(x))))


def _raw_log_dp(x: np.ndarray, p: float = 0.5) -> float:
    return float(np.log(np.sum(x**p)) / p)


def _raw_grad_log_s(x: np.ndarray) -> np.ndarray:
    s = -float(np.sum(x * np.log(x)))
    return -(np.log(x) + 1.0) / s


def _raw_grad_log_dp(x: np.ndarray, p: float = 0.5) -> np.ndarray:
    return np.asarray(x ** (p - 1.0) / np.sum(x**p), dtype=float)


def test_generator_gradients_match_finite_difference() -> None:
    # Ambient partial derivatives — G is defined on a neighborhood of the
    # simplex, so perturbations are NOT renormalized.
    rng = np.random.default_rng(3)
    m = rng.uniform(0.2, 1.0, 5)
    m = m / m.sum()
    eps = 1e-6
    for raw_log_g, grad_fn in (
        (_raw_log_s, entropy_grad_log_g),
        (_raw_log_dp, lambda x: diversity_grad_log_g(x, 0.5)),
    ):
        analytic = grad_fn(m[None, :])[0]
        numeric = np.empty(5)
        for i in range(5):
            up = m.copy()
            dn = m.copy()
            up[i] += eps
            dn[i] -= eps
            numeric[i] = (raw_log_g(up) - raw_log_g(dn)) / (2 * eps)
        np.testing.assert_allclose(analytic, numeric, rtol=1e-4, atol=1e-6)


def test_generator_hessians_match_finite_difference() -> None:
    rng = np.random.default_rng(4)
    m = rng.uniform(0.2, 1.0, 4)
    m = m / m.sum()
    eps = 1e-5
    for raw_grad, hess_fn in (
        (_raw_grad_log_s, entropy_hess_log_g),
        (_raw_grad_log_dp, lambda x: diversity_hess_log_g(x, 0.5)),
    ):
        analytic = hess_fn(m[None, :])[0]
        numeric = np.empty((4, 4))
        for j in range(4):
            up = m.copy()
            dn = m.copy()
            up[j] += eps
            dn[j] -= eps
            numeric[:, j] = (raw_grad(up) - raw_grad(dn)) / (2 * eps)
        np.testing.assert_allclose(analytic, numeric, rtol=1e-3, atol=1e-4)


# --- excess growth and relative returns --------------------------------------


def test_excess_growth_market_is_half_trace() -> None:
    # Under the relative covariance τ = σ^{logμ}, μᵀτμ = 0 (Σ dμ = 0), so
    # the market portfolio's own excess growth collapses to
    # γ*_μ = (1/2)Σ_i μ_i τ_ii — strictly positive in a diverse market,
    # and the quantity FG portfolios harvest.
    mu = market_weights(_market(n_steps=100))
    tau = log_weight_covariation(mu, DT)
    gamma_market = excess_growth_rate(mu[:-1], tau)
    half_tr = 0.5 * np.einsum("ti,tii->t", mu[:-1], tau)
    np.testing.assert_allclose(gamma_market, half_tr, atol=1e-3)
    assert float(np.mean(gamma_market)) > 0.0


def test_excess_growth_numeraire_invariance() -> None:
    caps = _market(n_steps=100)
    mu = market_weights(caps)
    pi = diversity_weights(mu, 0.5)
    g_x = excess_growth_rate(pi[:-1], log_cap_covariation(caps, DT))
    g_mu = excess_growth_rate(pi[:-1], log_weight_covariation(mu, DT))
    np.testing.assert_allclose(g_x, g_mu, atol=1e-10)


def test_relative_log_returns_single_stock_identity() -> None:
    # π = e_i (hold one stock): cumulative relative return = Δlog μ_i exactly.
    caps = _market(n_steps=100)
    mu = market_weights(caps)
    pi = np.zeros_like(mu)
    pi[:, 2] = 1.0
    rel = np.cumsum(relative_log_returns(caps, pi))
    expected = np.log(mu[1:, 2]) - np.log(mu[0, 2])
    np.testing.assert_allclose(rel, expected, atol=1e-10)


def test_relative_log_returns_market_is_zero() -> None:
    caps = _market(n_steps=100)
    mu = market_weights(caps)
    rel = relative_log_returns(caps, mu)
    np.testing.assert_allclose(rel, 0.0, atol=1e-12)


def test_relative_log_returns_rejects_mismatch() -> None:
    caps = _market(n_steps=50)
    mu = market_weights(caps)
    with pytest.raises(ValueError):
        relative_log_returns(caps, mu[:-2])


# --- master equation ---------------------------------------------------------


def test_master_equation_residual_small_diversity() -> None:
    caps = _market(n_assets=16, n_steps=2016)
    decomp = diversity_master_equation(caps, 0.5, DT)
    assert decomp.residual_rel < 0.10
    assert decomp.theta_integral > 0.0
    np.testing.assert_allclose(
        decomp.lhs, decomp.delta_log_g + decomp.theta_integral, atol=decomp.residual_abs + 1e-12
    )


def test_master_equation_residual_small_entropy() -> None:
    caps = _market(n_assets=16, n_steps=2016)
    decomp = entropy_master_equation(caps, DT)
    assert decomp.residual_rel < 0.10


def test_master_equation_telescopes_log_g() -> None:
    caps = _market(n_steps=200)
    mu = market_weights(caps)
    decomp = fg_master_equation(
        caps,
        diversity_weights(mu, 0.5),
        diversity_log_g(mu, 0.5),
        diversity_hess_log_g(mu, 0.5),
        DT,
    )
    direct = float(diversity_log_g(mu[-1:], 0.5)[0] - diversity_log_g(mu[:1], 0.5)[0])
    assert decomp.delta_log_g == pytest.approx(direct, abs=1e-12)


def test_diversity_drift_matches_general_formula() -> None:
    caps = _market(n_steps=300)
    mu = market_weights(caps)
    pi = diversity_weights(mu, 0.5)
    sigma_mu = log_weight_covariation(mu, DT)
    closed = diversity_drift_rate(pi[:-1], sigma_mu, 0.5)
    general = fg_drift_rate(
        pi[:-1],
        mu[:-1],
        diversity_hess_log_g(mu, 0.5)[:-1],
        sigma_mu,
        log_cap_covariation(caps, DT),
    )
    np.testing.assert_allclose(closed, general, atol=1e-10)


def test_diversity_theta_nonnegative_mean() -> None:
    # Θ_{D_p} = (1−p)/2 · Σ π_i τ_ii ≥ 0 in expectation; the realized
    # single-step rate is noisy but the cumulative drift must be positive
    # on a typical path.
    caps = _market(n_steps=1000)
    mu = market_weights(caps)
    pi = diversity_weights(mu, 0.5)
    theta = diversity_drift_rate(pi[:-1], log_weight_covariation(mu, DT), 0.5)
    assert float(np.sum(theta)) > 0.0


def test_master_equation_fail_closed() -> None:
    caps = _market(n_steps=100)
    mu = market_weights(caps)
    with pytest.raises(ValueError):
        fg_master_equation(
            caps, diversity_weights(mu, 0.5), np.array([0.0]), diversity_hess_log_g(mu, 0.5), DT
        )
    with pytest.raises(ValueError):
        fg_master_equation(
            caps,
            diversity_weights(mu, 0.5),
            diversity_log_g(mu, 0.5),
            diversity_hess_log_g(mu, 0.5)[:, :-1],
            DT,
        )
    with pytest.raises(ValueError):
        fg_master_equation(
            caps,
            diversity_weights(mu, 0.5),
            diversity_log_g(mu, 0.5),
            diversity_hess_log_g(mu, 0.5),
            -1.0,
        )


# --- local times --------------------------------------------------------------


def test_local_time_estimator_vs_exact_reflection() -> None:
    dt, n_steps = 2.5e-5, 60_000
    rng = np.random.default_rng(11)
    est_sum = true_sum = 0.0
    for _ in range(6):
        reflected, true_lt = reflected_path(0.0, rng.normal(0.0, np.sqrt(dt), n_steps))
        est_sum += local_time_at_zero(reflected, dt, 4.0 * np.sqrt(dt))
        true_sum += float(true_lt[-1])
    assert abs(est_sum - true_sum) / true_sum < 0.25


def test_local_time_zero_for_never_touching_path() -> None:
    # A path bounded away from zero accrues no local time.
    x = np.linspace(0.5, 0.8, 500)
    assert local_time_at_zero(x, 0.01, bandwidth=0.05) == pytest.approx(0.0)


def test_local_time_fail_closed() -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        local_time_at_zero(np.array([0.5, -0.2, 0.4]), 0.01)
    with pytest.raises(ValueError):
        local_time_at_zero(np.array([0.1]), 0.01)
    with pytest.raises(ValueError):
        local_time_at_zero(np.array([0.1, 0.2]), 0.01, bandwidth=-1.0)
    with pytest.raises(ValueError, match="degenerate"):
        local_time_at_zero(np.full(10, 0.3), 0.01)


def test_reflected_path_exact_compensator() -> None:
    rng = np.random.default_rng(2)
    eps = rng.normal(0.0, 0.01, 5000)
    r, lt = reflected_path(0.0, eps)
    assert np.all(r >= 0.0)
    assert np.all(np.diff(lt) >= 0.0)
    # L only pushes when the unreflected update would go below zero.
    y = r[:-1] + eps
    np.testing.assert_allclose(r[1:], np.maximum(y, 0.0))
    np.testing.assert_allclose(np.diff(lt), np.maximum(-y, 0.0))


def test_rank_gap_local_times_shape_and_nonneg() -> None:
    mu = market_weights(_market(n_assets=10, n_steps=400))
    lts = rank_gap_local_times(mu, DT)
    assert lts.shape == (9,)
    assert np.all(lts >= 0.0)
    assert np.isfinite(lts).all()


def test_reflected_path_fail_closed() -> None:
    with pytest.raises(ValueError):
        reflected_path(-0.5, np.array([0.1, 0.2]))
    with pytest.raises(ValueError):
        reflected_path(0.0, np.array([np.nan]))


# --- arbitrage horizon & growth-optimal --------------------------------------


def test_arbitrage_horizon_formula() -> None:
    mu0 = np.array([[0.6, 0.2, 0.2]])
    d0 = diversity_index(mu0, 0.5)[0]
    theta = 0.05
    expected = np.log(d0) / theta
    assert diversity_arbitrage_horizon(mu0, 0.5, theta) == pytest.approx(expected)


def test_arbitrage_horizon_monotone_in_drift_floor() -> None:
    mu0 = np.array([[0.4, 0.3, 0.3]])
    lo = diversity_arbitrage_horizon(mu0, 0.5, 0.02)
    hi = diversity_arbitrage_horizon(mu0, 0.5, 0.10)
    assert lo > hi


def test_arbitrage_horizon_fail_closed() -> None:
    mu0 = np.array([[0.5, 0.5]])
    with pytest.raises(ValueError):
        diversity_arbitrage_horizon(mu0, 0.5, 0.0)
    with pytest.raises(ValueError):
        diversity_arbitrage_horizon(mu0, 0.5, -1.0)
    with pytest.raises(ValueError):
        diversity_arbitrage_horizon(mu0, 1.5, 0.05)
    with pytest.raises(ValueError):
        diversity_arbitrage_horizon(np.array([[0.5, 0.5], [0.4, 0.6]]), 0.5, 0.05)


def test_first_passage_time() -> None:
    v = np.array([0.01, 0.03, -0.02, 0.05])
    assert first_passage_time(v, 0.5, 0.03) == pytest.approx(1.0)
    assert np.isnan(first_passage_time(v, 0.5, 10.0))
    with pytest.raises(ValueError):
        first_passage_time(np.array([np.inf]), 0.5)


def test_growth_optimal_two_asset_diagonal() -> None:
    # Diagonal sigma: γ* = (σ1+σ2)·π(1−π)/2 is maximized at the half-half mix.
    sigma = np.diag([0.2, 0.05])
    w = growth_optimal_weights(sigma)
    np.testing.assert_allclose(w, [0.5, 0.5], atol=1e-3)


def test_growth_optimal_beats_uniform() -> None:
    rng = np.random.default_rng(6)
    a = rng.normal(0.0, 1.0, (5, 5))
    sigma = a @ a.T / 5 + 0.1 * np.eye(5)
    w = growth_optimal_weights(sigma)
    assert w.sum() == pytest.approx(1.0)
    assert np.all(w >= -1e-9)
    g_opt = excess_growth_rate(w[None, :], sigma[None])[0]
    g_unif = excess_growth_rate(np.full((1, 5), 0.2), sigma[None])[0]
    assert g_opt >= g_unif - 1e-10


def test_growth_optimal_fail_closed() -> None:
    with pytest.raises(ValueError, match="symmetric"):
        growth_optimal_weights(np.array([[1.0, 0.9], [0.1, 1.0]]))
    with pytest.raises(ValueError, match="semidefinite"):
        growth_optimal_weights(np.array([[1.0, 2.0], [2.0, 1.0]]))


def test_arbitrage_diagnostics_consistency() -> None:
    caps = _market(n_steps=800)
    diag = arbitrage_diagnostics(caps, 0.5, DT)
    assert diag["theta_mean"] > 0.0
    assert diag["horizon_bound"] == pytest.approx(diag["log_diversity_start"] / diag["theta_mean"])
    if np.isfinite(diag["horizon_realized"]):
        assert diag["horizon_slack"] == pytest.approx(
            diag["horizon_bound"] / diag["horizon_realized"]
        )


# --- simulation & bench --------------------------------------------------------


def test_simulate_market_deterministic() -> None:
    a = simulate_gbm_market(n_assets=6, n_steps=50, dt=DT, seed=42)
    b = simulate_gbm_market(n_assets=6, n_steps=50, dt=DT, seed=42)
    np.testing.assert_array_equal(a, b)
    c = simulate_gbm_market(n_assets=6, n_steps=50, dt=DT, seed=43)
    assert not np.array_equal(a, c)


def test_simulate_market_fail_closed() -> None:
    with pytest.raises(ValueError):
        simulate_gbm_market(n_assets=1)
    with pytest.raises(ValueError):
        simulate_gbm_market(n_assets=5, n_steps=1)
    with pytest.raises(ValueError):
        simulate_gbm_market(n_assets=5, seed="x")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        simulate_gbm_market(n_assets=5, rho=1.2)


def test_bench_flat_float_dict() -> None:
    out = bench_fernholz_spt(24)
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) for v in out.values())
    assert all(np.isfinite(v) or k == "synthetic_horizon_slack" for k, v in out.items())


def test_bench_deterministic() -> None:
    assert bench_fernholz_spt(24) == bench_fernholz_spt(24)
    assert bench_fernholz_spt(24) != bench_fernholz_spt(25)


def test_bench_headline_accuracy() -> None:
    out = bench_fernholz_spt(24)
    assert out["synthetic_diversity_master_eq_rel_residual"] < 0.10
    assert out["synthetic_entropy_master_eq_rel_residual"] < 0.10
    assert out["synthetic_theta_closed_form_gap"] < 1e-8
    assert out["synthetic_local_time_rel_error"] < 0.20
    assert out["synthetic_entropy_uniform_error"] < 1e-10
    assert out["synthetic_diversity_theta_integral"] > 0.0
