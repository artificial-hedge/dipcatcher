"""Tests for Wasserstein-DRO mean-variance portfolios."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.portfolio.wasserstein_dro import DroResult, solve_wasserstein_dro


def _problem(seed: int = 7, n: int = 6) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(n, n))
    cov = a.T @ a / n + np.eye(n) * 0.5
    mu = rng.uniform(0.02, 0.10, size=n)
    return mu, cov


def _nominal_solve(
    mu: np.ndarray,
    cov: np.ndarray,
    risk_aversion: float = 1.0,
) -> np.ndarray:
    import cvxpy as cp

    n = mu.size
    w = cp.Variable(n)
    prob = cp.Problem(
        cp.Minimize(-mu @ w + risk_aversion * cp.quad_form(w, cov)),
        [cp.sum(w) == 1.0, w >= 0.0],
    )
    prob.solve(solver="CLARABEL", verbose=False)
    assert prob.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}
    return np.asarray(w.value, dtype=float).reshape(-1)


class TestRadiusZero:
    def test_matches_nominal_mean_variance(self):
        mu, cov = _problem()
        res = solve_wasserstein_dro(mu, cov, radius=0.0)
        w_nom = _nominal_solve(mu, cov)
        assert isinstance(res, DroResult)
        assert np.allclose(res.weights, w_nom, atol=1e-6)
        assert res.dual_lambda == 0.0
        assert res.worst_case_objective == pytest.approx(res.nominal_objective, abs=1e-9)

    def test_matches_nominal_with_cap(self):
        mu, cov = _problem()
        res = solve_wasserstein_dro(mu, cov, radius=0.0, per_asset_cap=0.4)
        w_nom = _nominal_solve(mu, cov)
        assert np.allclose(res.weights, w_nom, atol=1e-6)
        assert np.all(res.weights <= 0.4 + 1e-8)


class TestShrinkage:
    def test_radius_monotone_shrinks_norm(self):
        mu, cov = _problem()
        w0 = solve_wasserstein_dro(mu, cov, radius=0.0).weights
        w1 = solve_wasserstein_dro(mu, cov, radius=0.1).weights
        w2 = solve_wasserstein_dro(mu, cov, radius=0.5).weights
        assert np.linalg.norm(w1) <= np.linalg.norm(w0) + 1e-9
        assert np.linalg.norm(w2) <= np.linalg.norm(w1) + 1e-9
        # Risk shrinkage is strict for this problem: radius 0.5 must move.
        assert np.linalg.norm(w2) < np.linalg.norm(w1)

    def test_toward_equal_weights(self):
        mu, cov = _problem()
        w = solve_wasserstein_dro(mu, cov, radius=1.0e4).weights
        n = mu.size
        assert np.allclose(w, np.full(n, 1.0 / n), atol=1e-4)


class TestConstraints:
    def test_long_only_budget_exact(self):
        mu, cov = _problem()
        res = solve_wasserstein_dro(mu, cov, radius=0.2)
        assert np.isclose(res.weights.sum(), 1.0, atol=1e-8)
        assert np.all(res.weights >= -1e-9)
        assert np.all(res.weights <= 1.0 + 1e-9)

    def test_budget_scaling(self):
        mu, cov = _problem()
        res = solve_wasserstein_dro(mu, cov, radius=0.2, budget=2.0)
        assert np.isclose(res.weights.sum(), 2.0, atol=1e-8)
        assert np.all(res.weights >= -1e-9)

    def test_per_asset_cap_respected(self):
        mu, cov = _problem(n=8)
        cap = 0.2
        res = solve_wasserstein_dro(mu, cov, radius=0.3, per_asset_cap=cap)
        assert np.all(res.weights <= cap + 1e-8)
        assert np.isclose(res.weights.sum(), 1.0, atol=1e-8)

    def test_short_side_budget(self):
        mu, cov = _problem()
        res = solve_wasserstein_dro(mu, cov, radius=0.05, long_only=False)
        assert np.isclose(res.weights.sum(), 1.0, atol=1e-8)


class TestWorstCaseObjective:
    def test_worst_case_ge_nominal(self):
        mu, cov = _problem()
        for radius in (0.0, 0.05, 0.2, 1.0):
            res = solve_wasserstein_dro(mu, cov, radius=radius)
            assert res.worst_case_objective >= res.nominal_objective - 1e-12
            expected = res.nominal_objective + radius * np.linalg.norm(res.weights)
            assert res.worst_case_objective == pytest.approx(expected, rel=1e-9)

    def test_worst_case_grows_with_radius(self):
        mu, cov = _problem()
        r1 = solve_wasserstein_dro(mu, cov, radius=0.1)
        r2 = solve_wasserstein_dro(mu, cov, radius=0.5)
        assert r2.worst_case_objective >= r1.worst_case_objective - 1e-12


class TestFailClosed:
    def test_non_psd_cov(self):
        mu, _ = _problem()
        bad = np.diag([1.0, 1.0, 1.0, 1.0, 1.0, -0.5])
        with pytest.raises(ValueError, match="positive semidefinite"):
            solve_wasserstein_dro(mu, bad, radius=0.1)

    def test_negative_radius(self):
        mu, cov = _problem()
        with pytest.raises(ValueError, match="radius"):
            solve_wasserstein_dro(mu, cov, radius=-0.1)

    def test_dim_mismatch(self):
        _, cov = _problem()
        with pytest.raises(ValueError, match="matching cov"):
            solve_wasserstein_dro(np.zeros(4), cov, radius=0.1)

    def test_non_positive_budget(self):
        mu, cov = _problem()
        with pytest.raises(ValueError, match="budget"):
            solve_wasserstein_dro(mu, cov, radius=0.1, budget=0.0)

    def test_non_positive_risk_aversion(self):
        mu, cov = _problem()
        with pytest.raises(ValueError, match="risk_aversion"):
            solve_wasserstein_dro(mu, cov, radius=0.1, risk_aversion=0.0)

    def test_non_optimal_status_raises_runtime_error(self, monkeypatch):
        import cvxpy as cp

        real_solve = cp.Problem.solve

        def fake_solve(self, *args, **kwargs):
            real_solve(self, *args, **kwargs)
            monkeypatch.setattr(type(self), "status", property(lambda _s: cp.INFEASIBLE))

        monkeypatch.setattr(cp.Problem, "solve", fake_solve)
        mu, cov = _problem()
        with pytest.raises(RuntimeError, match="solver status"):
            solve_wasserstein_dro(mu, cov, radius=0.1)

    def test_solver_exception_raises_runtime_error(self, monkeypatch):
        import cvxpy as cp

        def boom(self, *args, **kwargs):
            # SolverError is cvxpy's real solver-failure exception; the
            # module's guard is narrowed to (SolverError, ArithmeticError,
            # ValueError) per the quality ratchet — exotic exceptions
            # propagate unwrapped (still fail-closed).
            raise cp.error.SolverError("boom")

        monkeypatch.setattr(cp.Problem, "solve", boom)
        mu, cov = _problem()
        with pytest.raises(RuntimeError, match="solver error"):
            solve_wasserstein_dro(mu, cov, radius=0.1)

    def test_tiny_cap_rejected_early(self):
        mu, cov = _problem()
        with pytest.raises(ValueError, match="too small"):
            solve_wasserstein_dro(mu, cov, radius=0.1, per_asset_cap=0.05)


class TestDeterminism:
    def test_identical_across_solves(self):
        mu, cov = _problem()
        r1 = solve_wasserstein_dro(mu, cov, radius=0.25, per_asset_cap=0.5)
        r2 = solve_wasserstein_dro(mu, cov, radius=0.25, per_asset_cap=0.5)
        np.testing.assert_array_equal(r1.weights, r2.weights)
        assert r1.nominal_objective == r2.nominal_objective
        assert r1.worst_case_objective == r2.worst_case_objective
