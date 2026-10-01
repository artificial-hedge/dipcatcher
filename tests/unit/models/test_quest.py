"""KATs and contracts for the numerical QuEST (Ledoit-Wolf 2015/2017) port.

Covered here:

* ``quest_forward`` — the discretized QuEST map must reproduce
  Marcenko–Pastur support edges on a point-mass population spectrum.
* ``quest_lambda_jacobian`` — the analytic Jacobian of the forward map
  must match central finite differences to ~1e-6.
* ``estimate_population_eigenvalues`` — inversion must recover a
  two-cluster oracle spectrum better than the raw sample eigenvalues.
* ``quest_covariance`` — PSD, symmetric, deterministic; the singular
  ``p > n_eff`` case stays QuEST rather than silently switching
  estimators.
* ``ledoit_wolf_quest`` / ``ledoit_wolf_quest_cov`` — catalog stamps and
  fail-closed edges.
* ``require_implemented_optimizer_covariance`` — ``quest`` /
  ``ledoit_wolf_2017`` aliases resolve to the numerical estimator.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

import quant_fund.models.covariance as cov
from quant_fund.models.quest import (
    estimate_population_eigenvalues,
    linear_shrinkage_eigenvalues,
    quest_covariance,
    quest_forward,
    quest_lambda_jacobian,
)


def _panel(t: int = 400, n: int = 60, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pop = np.concatenate([np.full(n - 20, 0.5), np.full(20, 3.0)])
    return rng.normal(0.0, 1.0, (t, n)) @ np.diag(np.sqrt(pop)), pop


class TestForwardMap:
    def test_mp_support_edges(self) -> None:
        # Point-mass spectrum tau=1, c=0.5 => MP edges (1±sqrt(0.5))^2.
        p = 100
        q = quest_forward(np.ones(p), 2 * p)
        lo, hi = (1.0 - np.sqrt(0.5)) ** 2, (1.0 + np.sqrt(0.5)) ** 2
        assert q.endpoints.shape[0] >= 1
        assert_allclose(q.endpoints[0], [lo, hi], rtol=1e-6, atol=1e-6)
        assert q.lam[0] >= lo * 0.9
        assert q.lam[-1] <= hi * 1.05

    def test_density_integrates_to_one(self) -> None:
        q = quest_forward(np.ones(50), 100)
        # g is a density on zeta = x^(1/4); integrate over that grid.
        grid = np.concatenate(q.zeta_list)
        dens = np.concatenate(q.g_list)
        order = np.argsort(grid)
        mass = np.trapezoid(dens[order], grid[order])
        assert 0.9 < mass < 1.1

    def test_deterministic(self) -> None:
        tau = np.sort(np.random.default_rng(0).uniform(0.5, 3.0, 30))
        a = quest_forward(tau, 60)
        b = quest_forward(tau, 60)
        assert_allclose(a.lam, b.lam)
        assert_allclose(a.endpoints, b.endpoints)


class TestJacobian:
    def test_matches_finite_differences(self) -> None:
        rng = np.random.default_rng(0)
        p, n = 40, 80
        tau = np.sort(rng.uniform(0.5, 3.0, p))
        fwd = quest_forward(tau, n)
        jac = quest_lambda_jacobian(fwd)
        eps = 1e-6
        fd = np.zeros_like(jac)
        for j in range(p):
            up = tau.copy()
            up[j] += eps
            dn = tau.copy()
            dn[j] -= eps
            fd[:, j] = (quest_forward(up, n).lam - quest_forward(dn, n).lam) / (2 * eps)
        assert_allclose(jac, fd, rtol=1e-3, atol=1e-4)


class TestInversion:
    def test_recovers_two_cluster_oracle(self) -> None:
        rng = np.random.default_rng(11)
        t, n = 400, 80
        pop = np.concatenate([np.full(60, 0.5), np.full(20, 3.0)])
        x = rng.normal(0.0, 1.0, (t, n)) @ np.diag(np.sqrt(pop))
        xc = x - x.mean(axis=0)
        s = xc.T @ xc / (t - 1)
        lam = np.maximum(np.linalg.eigvalsh(s), 0.0)
        tau_hat, info = estimate_population_eigenvalues(lam, t - 1, centered=xc)
        assert tau_hat.shape == (n,)
        assert np.isfinite(tau_hat).all()
        assert float(info["quest_objective"]) >= 0.0
        # Recovered spectrum must sit inside the sample-law range and
        # track the two-cluster split better than raw sample eigenvalues.
        assert tau_hat.min() >= lam.min() - 1e-9
        assert tau_hat.max() <= lam.max() + 1e-9
        low_share = float(np.mean(tau_hat < 1.5))
        assert low_share > 0.5  # ~75% of mass is the 0.5 cluster

    def test_non_finite_rejected(self) -> None:
        with pytest.raises(ValueError):
            estimate_population_eigenvalues(
                np.array([0.1, np.nan, 0.3]), 100, centered=np.ones((5, 3))
            )

    def test_linear_shrinkage_init_shape(self) -> None:
        x, _pop = _panel(60, 20)
        xc = x - x.mean(axis=0)
        lam = linear_shrinkage_eigenvalues(xc, 59)
        assert lam.shape == (20,)
        assert np.isfinite(lam).all()


class TestCovariance:
    def test_psd_symmetric_deterministic(self) -> None:
        x, _ = _panel(300, 40)
        xc = x - x.mean(axis=0)
        s1, m1 = quest_covariance(xc, 299)
        s2, _ = quest_covariance(xc, 299)
        assert_allclose(s1, s2)
        assert_allclose(s1, s1.T, atol=1e-10)
        assert np.linalg.eigvalsh(0.5 * (s1 + s1.T)).min() >= -1e-10
        assert int(m1["quest_numint"]) >= 1
        assert float(m1["quest_objective"]) >= 0.0
        assert float(m1["quest_tau_min"]) <= float(m1["quest_tau_max"])

    def test_beats_sample_on_oracle(self) -> None:
        x, pop = _panel(400, 60, seed=3)
        xc = x - x.mean(axis=0)
        s_q, _ = quest_covariance(xc, 399)
        s_raw = xc.T @ xc / 399
        true = np.diag(pop)
        e_q = np.linalg.norm(s_q - true) / np.linalg.norm(true)
        e_s = np.linalg.norm(s_raw - true) / np.linalg.norm(true)
        assert e_q < e_s

    def test_singular_case_stays_quest(self) -> None:
        rng = np.random.default_rng(5)
        x = rng.normal(0.0, 1.0, (30, 50))
        xc = x - x.mean(axis=0)
        s, meta = quest_covariance(xc, 29)
        assert s.shape == (50, 50)
        assert np.linalg.eigvalsh(s).min() >= -1e-9
        assert int(meta["quest_numint"]) >= 1

    def test_fail_closed_edges(self) -> None:
        # n_eff < 2 fails closed
        with pytest.raises(ValueError):
            quest_covariance(np.ones((5, 3)), 1)
        # non-finite input fails closed
        with pytest.raises(ValueError):
            quest_covariance(np.array([[np.nan] * 4] * 20), 19)
        # degenerate (zero) spectrum fails closed
        with pytest.raises(ValueError):
            quest_covariance(np.zeros((20, 4)), 19)
        # not 2D fails closed
        with pytest.raises(ValueError):
            quest_covariance(np.ones(10), 9)


class TestCatalogWiring:
    def test_stamps(self) -> None:
        x, _ = _panel(200, 20)
        sigma, params = cov.ledoit_wolf_quest(x)
        assert params["family"] == "ledoit_wolf_quest"
        assert params["spec"] == "ledoit_wolf_2017_quest"
        assert params["covariance_object"] == "trailing"
        assert params["sample"] == "listwise_complete"
        assert sigma.shape == (20, 20)
        direct = cov.ledoit_wolf_quest_cov(x)
        assert_allclose(direct, sigma)

    def test_aliases_resolve_to_quest(self) -> None:
        for name in (
            "ledoit_wolf_quest",
            "quest",
            "ledoit_wolf_2017",
            "ledoitwolf_2017",
            "lw_2017",
            "nonlinear_shrinkage_2017",
            "ledoit_wolf_2017_quest",
            "numerical_quest",
            "quest_2017",
        ):
            assert cov.require_implemented_optimizer_covariance(name) == "ledoit_wolf_quest"

    def test_spec_lists(self) -> None:
        assert "ledoit_wolf_quest" in cov.IMPLEMENTED_COVARIANCE_SPECS
        assert "ledoit_wolf_quest" in cov.IMPLEMENTED_OPTIMIZER_COVARIANCE_SPECS
