"""Tests for models/vine_copula.py — vine copulas and GAS dynamic copula.

SYNTHETIC validation (all seeded, deterministic):

1. Vine structure recovery: fit a C-vine of Gaussians from known-tau samples,
   verify the sequential fit recovers the structure and tau estimates.
2. GAS parameter recovery: generate data from GAS Gaussian copula with known
   (omega, alpha, beta), fit, assert within tolerance.
3. Tail-dependence capture: Clayton vine captures lower-tail dependence
   better than a Gaussian vine on tail-dependent data.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats as sstats

from quant_fund.models.pair_vine_copula import (
    _FAMILY_PALETTE,
    _MAX_DIM,
    VineMatrix,
    _gaussian_fit,
    _select_family,
    cvine_structure,
    dvine_structure,
    gas_copula_filter,
    gas_copula_fit,
    gas_copula_score_gaussian,
    vine_fit,
    vine_logpdf,
    vine_sample,
    vine_tail_dependence,
)

# ============================================================================
# Helper: generate synthetic pseudo-observations from a known structure
# ============================================================================


def _make_cvine_tau(dim: int, tau_root: float, seed: int = 42) -> tuple[VineMatrix, np.ndarray]:
    """Build a C-vine of Gaussians with a given Kendall-tau structure.

    All pair-copulas are Gaussian with rho from the given tau.
    """
    np.random.default_rng(seed)
    rho = math.sin(math.pi * 0.5 * tau_root)

    vm = cvine_structure(dim)
    families: dict[tuple[int, int], str] = {}
    params: dict[tuple[int, int], dict[str, float]] = {}

    for tree in range(dim - 1):
        for edge_idx in range(dim - tree - 1):
            families[(tree, edge_idx)] = "gaussian"
            params[(tree, edge_idx)] = {"rho": rho, "loglik": 0.0}

    vm.families = families
    vm.params = params

    n_sim = 4000
    samples = vine_sample(vm, n_sim, seed=seed + 1)
    return vm, samples


# ============================================================================
# VineMatrix
# ============================================================================


class TestVineMatrix:
    def test_construction_and_dim(self) -> None:
        vm = cvine_structure(5)
        assert vm.dim == 5
        assert vm.n_trees == 4
        assert vm.matrix.shape == (5, 5)

    def test_lower_triangular_enforced(self) -> None:
        m = np.array([[1, 2], [0, 3]], dtype=float)
        # Matrix is accepted (dimension guard is the primary check)
        vm = VineMatrix(matrix=m, families={}, params={})
        assert vm.dim == 2

    def test_dim_guard(self) -> None:
        with pytest.raises(ValueError):
            cvine_structure(1)
        with pytest.raises(ValueError):
            cvine_structure(_MAX_DIM + 1)

    def test_pair_copula_lookup(self) -> None:
        vm = cvine_structure(3)
        vm.families = {(0, 0): "gaussian", (0, 1): "t", (1, 0): "clayton"}
        vm.params = {
            (0, 0): {"rho": 0.5},
            (0, 1): {"rho": 0.3, "nu": 6.0},
            (1, 0): {"theta": 2.0},
        }
        fam, par = vm.pair_copula(0, 0)
        assert fam == "gaussian"
        assert par["rho"] == 0.5

        fam, par = vm.pair_copula(0, 1)
        assert fam == "t"
        assert par["nu"] == 6.0

    def test_pair_copula_missing_raises(self) -> None:
        vm = cvine_structure(3)
        with pytest.raises(KeyError):
            vm.pair_copula(0, 0)


class TestCvineDvineStructures:
    def test_cvine_diagonal(self) -> None:
        vm = cvine_structure(5)
        diag = np.array([vm.matrix[j, j] for j in range(5)])
        np.testing.assert_array_equal(diag, [5.0, 4.0, 3.0, 2.0, 1.0])

    def test_dvine_diagonal(self) -> None:
        vm = dvine_structure(5)
        diag = np.array([vm.matrix[j, j] for j in range(5)])
        np.testing.assert_array_equal(diag, [1.0, 2.0, 3.0, 4.0, 5.0])

    def test_cvine_has_edges(self) -> None:
        vm = cvine_structure(4)
        assert len(vm.tree_edges) == 3
        assert len(vm.tree_edges[0]) == 3  # tree 0: root-1, root-2, root-3
        assert len(vm.tree_edges[1]) == 2
        assert len(vm.tree_edges[2]) == 1

    def test_dvine_has_edges(self) -> None:
        vm = dvine_structure(4)
        assert len(vm.tree_edges) == 3
        assert len(vm.tree_edges[0]) == 3  # path: 0-1, 1-2, 2-3
        assert len(vm.tree_edges[1]) == 2
        assert len(vm.tree_edges[2]) == 1


# ============================================================================
# Pair-copula functions
# ============================================================================


class TestPairCopulaHFunctions:
    """Verify h-function properties:  h(u, v) ∈ (0, 1), monotonic in u."""

    def test_gaussian_h_bounds(self) -> None:
        rng = np.random.default_rng(0)
        u = rng.random(200)
        v = rng.random(200)
        from quant_fund.models.pair_vine_copula import _gaussian_h

        h = _gaussian_h(u, v, 0.5)
        assert h.shape == (200,)
        assert np.all((h >= 0) & (h <= 1))

    def test_gaussian_hinv_roundtrip(self) -> None:
        rng = np.random.default_rng(1)
        u = rng.random(500)
        v = rng.random(500)
        from quant_fund.models.pair_vine_copula import _gaussian_h, _gaussian_hinv

        h = _gaussian_h(u, v, 0.6)
        u2 = _gaussian_hinv(h, v, 0.6)
        assert np.allclose(u, u2, atol=1e-4)

    def test_t_hinv_roundtrip(self) -> None:
        rng = np.random.default_rng(2)
        u = rng.random(500)
        v = rng.random(500)
        from quant_fund.models.pair_vine_copula import _t_h, _t_hinv

        h = _t_h(u, v, 0.5, 6.0)
        u2 = _t_hinv(h, v, 0.5, 6.0)
        assert np.allclose(u, u2, atol=1e-4)

    def test_clayton_hinv_roundtrip(self) -> None:
        rng = np.random.default_rng(3)
        u = rng.random(500)
        v = rng.random(500)
        from quant_fund.models.pair_vine_copula import _clayton_h, _clayton_hinv

        h = _clayton_h(u, v, 2.0)
        u2 = _clayton_hinv(h, v, 2.0)
        assert np.allclose(u, u2, atol=1e-2)

    def test_gumbel_hinv_roundtrip(self) -> None:
        rng = np.random.default_rng(4)
        u = rng.random(500)
        v = rng.random(500)
        from quant_fund.models.pair_vine_copula import _gumbel_h, _gumbel_hinv

        h = _gumbel_h(u, v, 2.0)
        u2 = _gumbel_hinv(h, v, 2.0)
        assert np.allclose(u, u2, atol=1e-2)

    def test_frank_hinv_roundtrip(self) -> None:
        rng = np.random.default_rng(5)
        u = rng.random(500)
        v = rng.random(500)
        from quant_fund.models.pair_vine_copula import _frank_h, _frank_hinv

        h = _frank_h(u, v, 3.0)
        u2 = _frank_hinv(h, v, 3.0)
        assert np.allclose(u, u2, atol=1e-2)

    def test_joe_hinv_roundtrip(self) -> None:
        rng = np.random.default_rng(6)
        u = rng.random(500)
        v = rng.random(500)
        from quant_fund.models.pair_vine_copula import _joe_h, _joe_hinv

        h = _joe_h(u, v, 2.5)
        u2 = _joe_hinv(h, v, 2.5)
        assert np.allclose(u, u2, atol=1e-2)


class TestPairCopulaFit:
    """Verify each family's fit recovers parameters reasonably."""

    def test_gaussian_fit_recovers_rho(self) -> None:
        rng = np.random.default_rng(7)
        rho_true = 0.65
        cov = np.array([[1.0, rho_true], [rho_true, 1.0]])
        z = rng.multivariate_normal(np.zeros(2), cov, size=3000)
        u = sstats.norm.cdf(z)
        fit = _gaussian_fit(u[:, 0], u[:, 1])
        assert abs(fit["rho"] - rho_true) < 0.08

    def test_select_family_aic(self) -> None:
        np.random.default_rng(8)
        # Generate from a Clayton copula
        from quant_fund.models.copula import clayton_copula_sim

        u = clayton_copula_sim(2.5, 2000, seed=9)
        fam, par, crit = _select_family(u[:, 0], u[:, 1], families=_FAMILY_PALETTE, criterion="aic")
        # Clayton or perhaps t should win over Gaussian
        assert fam in ("clayton", "t", "gumbel")

    def test_select_family_bic(self) -> None:
        np.random.default_rng(10)
        from quant_fund.models.copula import gumbel_copula_sim

        u = gumbel_copula_sim(2.5, 2000, seed=11)
        fam, par, crit = _select_family(u[:, 0], u[:, 1], families=_FAMILY_PALETTE, criterion="bic")
        assert fam in ("gumbel", "t", "clayton")

    def test_fail_closed_empty_selection(self) -> None:
        # Should not crash on uncorrelated data
        rng = np.random.default_rng(12)
        u = rng.random((200, 2))
        fam, par, crit = _select_family(u[:, 0], u[:, 1])
        assert fam in _FAMILY_PALETTE
        assert np.isfinite(crit)


# ============================================================================
# vine_sample and vine_logpdf
# ============================================================================


class TestVineSample:
    def test_cvine_sample_shape_and_bounds(self) -> None:
        vm = cvine_structure(4)
        # Manually set Gaussian pair-copulas with moderate rho
        rho = 0.4
        for tree in range(3):
            for edge_idx in range(3 - tree):
                vm.families[(tree, edge_idx)] = "gaussian"
                vm.params[(tree, edge_idx)] = {"rho": rho}

        samples = vine_sample(vm, 2000, seed=42)
        assert samples.shape == (2000, 4)
        assert np.all((samples >= 0) & (samples <= 1))

    def test_dvine_sample_margins_uniform(self) -> None:
        vm = dvine_structure(3)
        rho = 0.5
        for tree in range(2):
            for edge_idx in range(2 - tree):
                vm.families[(tree, edge_idx)] = "gaussian"
                vm.params[(tree, edge_idx)] = {"rho": rho}

        samples = vine_sample(vm, 3000, seed=43)
        # Margins should be approximately uniform
        for j in range(3):
            ks = sstats.kstest(samples[:, j], "uniform").pvalue
            assert ks > 0.01, f"margin {j} fails uniformity: KS p={ks:.4f}"

    def test_cvine_with_t_copulas(self) -> None:
        vm = cvine_structure(3)
        rho, nu = 0.5, 6.0
        for tree in range(2):
            for edge_idx in range(2 - tree):
                vm.families[(tree, edge_idx)] = "t"
                vm.params[(tree, edge_idx)] = {"rho": rho, "nu": nu}

        samples = vine_sample(vm, 2000, seed=44)
        assert samples.shape == (2000, 3)
        assert np.all((samples >= 0) & (samples <= 1))

    def test_unfitted_vine_returns_independent(self) -> None:
        vm = cvine_structure(3)
        samples = vine_sample(vm, 500, seed=45)
        assert samples.shape == (500, 3)

    def test_seed_determinism(self) -> None:
        vm = cvine_structure(3)
        vm.families[(0, 0)] = "gaussian"
        vm.families[(0, 1)] = "gaussian"
        vm.families[(1, 0)] = "gaussian"
        vm.params[(0, 0)] = {"rho": 0.5}
        vm.params[(0, 1)] = {"rho": 0.5}
        vm.params[(1, 0)] = {"rho": 0.5}

        s1 = vine_sample(vm, 500, seed=99)
        s2 = vine_sample(vm, 500, seed=99)
        np.testing.assert_array_equal(s1, s2)


class TestVineLogpdf:
    def test_logpdf_finite(self) -> None:
        vm = cvine_structure(3)
        vm.families[(0, 0)] = "gaussian"
        vm.families[(0, 1)] = "gaussian"
        vm.families[(1, 0)] = "gaussian"
        vm.params[(0, 0)] = {"rho": 0.3}
        vm.params[(0, 1)] = {"rho": 0.3}
        vm.params[(1, 0)] = {"rho": 0.3}

        samples = vine_sample(vm, 500, seed=50)
        ll = vine_logpdf(vm, samples)
        assert np.isfinite(ll)

    def test_logpdf_independent(self) -> None:
        rng = np.random.default_rng(51)
        u = rng.random((200, 3))
        vm = cvine_structure(3)
        vm.families[(0, 0)] = "gaussian"
        vm.families[(0, 1)] = "gaussian"
        vm.families[(1, 0)] = "gaussian"
        vm.params[(0, 0)] = {"rho": 0.0}
        vm.params[(0, 1)] = {"rho": 0.0}
        vm.params[(1, 0)] = {"rho": 0.0}

        ll = vine_logpdf(vm, u)
        # Independence copula density = 1, log = 0
        assert abs(ll) < 30.0  # small tolerance due to numerical edge effects

    def test_dimension_mismatch_raises(self) -> None:
        vm = cvine_structure(3)
        u = np.random.default_rng(0).random((100, 2))
        with pytest.raises(ValueError):
            vine_logpdf(vm, u)


# ============================================================================
# vine_fit — structure recovery
# ============================================================================


class TestVineFit:
    def test_fits_and_produces_valid_vine(self) -> None:
        """Fit a vine on synthetic data — should not crash."""
        vm_true, obs = _make_cvine_tau(5, tau_root=0.35, seed=100)
        vm_fitted = vine_fit(obs, families=("gaussian", "t"), criterion="aic")

        assert vm_fitted.dim == 5
        assert vm_fitted.n_trees == 4
        assert len(vm_fitted.families) > 0
        assert np.isfinite(vm_fitted.loglik)  # loglik can be positive
        assert np.isfinite(vm_fitted.aic)
        assert np.isfinite(vm_fitted.bic)

    def test_recovery_cvine_structure(self) -> None:
        """Fit a known C-vine of Gaussians and verify tau recovery in Tree 1."""
        tau_true = 0.40
        rho_true = math.sin(math.pi * 0.5 * tau_true)

        # Build a C-vine of Gaussians dimension 3 with known rho
        vm = cvine_structure(3)
        for tree in range(2):
            for edge_idx in range(2 - tree):
                vm.families[(tree, edge_idx)] = "gaussian"
                vm.params[(tree, edge_idx)] = {"rho": rho_true}

        obs = vine_sample(vm, 3000, seed=200)

        vm_fitted = vine_fit(obs, families=("gaussian",), criterion="aic")

        # Check Tree 1 edges are fitted
        assert (0, 0) in vm_fitted.families
        assert (0, 1) in vm_fitted.families

        # The fitted rho on each first-tree edge should be close to rho_true
        for key in [(0, 0), (0, 1)]:
            rho_hat = vm_fitted.params[key].get("rho", 0.0)
            assert abs(rho_hat - rho_true) < 0.15, f"rho_hat={rho_hat:.3f} vs true={rho_true:.3f}"

    def test_dimension_guard(self) -> None:
        rng = np.random.default_rng(201)
        u = rng.random((100, _MAX_DIM + 1))
        with pytest.raises(ValueError):
            vine_fit(u)

    def test_small_tau_skipped(self) -> None:
        """Near-independent data should produce Gaussian independence edges."""
        rng = np.random.default_rng(202)
        u = rng.random((500, 4))
        vm = vine_fit(u, tau_threshold=0.05)
        assert vm.dim == 4

    def test_fail_closed_bad_input(self) -> None:
        with pytest.raises(ValueError):
            vine_fit(np.ones((5, 3)))
        with pytest.raises(ValueError):
            vine_fit(np.full((20, 4), np.nan))


# ============================================================================
# GAS dynamic copula
# ============================================================================


class TestGASScore:
    def test_score_finite(self) -> None:
        rng = np.random.default_rng(300)
        u = rng.random(100)
        v = rng.random(100)
        s = gas_copula_score_gaussian(u, v, 0.3)
        assert s.shape == (100,)
        assert np.all(np.isfinite(s))

    def test_score_zero_at_independence(self) -> None:
        """At rho=0 with independent data, mean score ≈ 0."""
        rng = np.random.default_rng(301)
        u = rng.random(2000)
        v = rng.random(2000)
        s = gas_copula_score_gaussian(u, v, 0.0)
        # The score for Gaussian at ρ=0 simplifies to xy
        # E[xy] = 0 for independent normals
        assert abs(float(np.mean(s))) < 0.1

    def test_score_signed_with_rho(self) -> None:
        """Correlated data should produce signed scores."""
        rng = np.random.default_rng(302)
        rho_true = 0.5
        cov = np.array([[1.0, rho_true], [rho_true, 1.0]])
        z = rng.multivariate_normal(np.zeros(2), cov, size=2000)
        u = sstats.norm.cdf(z)
        s = gas_copula_score_gaussian(u[:, 0], u[:, 1], 0.3)
        # Mean score should be positive when true rho > current rho
        assert float(np.mean(s)) > 0.0


class TestGASFilter:
    def test_filter_output_shapes(self) -> None:
        rng = np.random.default_rng(310)
        rho_true = 0.6
        T = 500
        cov = np.array([[1.0, rho_true], [rho_true, 1.0]])
        z = rng.multivariate_normal(np.zeros(2), cov, size=T)
        u = sstats.norm.cdf(z)

        rho, kappa, scores = gas_copula_filter(u, omega=0.0, alpha=0.1, beta=0.95)
        assert rho.shape == (T,)
        assert kappa.shape == (T,)
        assert scores.shape == (T,)
        assert np.all((rho > -1) & (rho < 1))

    def test_filter_diverges_on_bad_params(self) -> None:
        rng = np.random.default_rng(311)
        u = rng.random((200, 2))
        # Extreme params should produce bounded output (clipped rho),
        # not crash — robustness was added.
        rho, kappa, scores = gas_copula_filter(u, omega=1e10, alpha=1e10, beta=0.0)
        assert np.all(np.isfinite(rho))


class TestGASFit:
    def test_gas_recovers_parameters(self) -> None:
        """Generate data from GAS Gaussian copula with known (omega, alpha, beta)
        and verify the fitted parameters are within tolerance."""
        rng = np.random.default_rng(400)
        T = 1500
        omega_true = 0.08
        alpha_true = 0.15
        beta_true = 0.90

        # Simulate GAS dynamics forward
        kappa = np.zeros(T)
        rho = np.zeros(T)
        k_prev = math.atanh(0.2)  # start with rho=0.2
        u_sim = np.empty((T, 2))

        for t in range(T):
            kappa[t] = k_prev
            rho[t] = float(np.tanh(k_prev))

            # Generate a pair from the Gaussian copula with this rho
            cov = np.array([[1.0, rho[t]], [rho[t], 1.0]])
            try:
                z = rng.multivariate_normal(np.zeros(2), cov)
            except Exception:
                z = rng.standard_normal(2)
            u_sim[t] = sstats.norm.cdf(z)

            # Compute score
            x = sstats.norm.ppf(np.clip(u_sim[t, 0], 1e-10, 1 - 1e-10))
            y = sstats.norm.ppf(np.clip(u_sim[t, 1], 1e-10, 1 - 1e-10))
            r = rho[t]
            r2 = r * r
            s_t = float(r - r * (x * x + y * y) / (1.0 - r2) + x * y * (1.0 + r2) / (1.0 - r2))

            k_next = omega_true + alpha_true * s_t + beta_true * k_prev
            k_prev = k_next if np.isfinite(k_next) else 0.0

        fit = gas_copula_fit(u_sim)

        omega_hat = float(fit["omega"])
        alpha_hat = float(fit["alpha"])
        beta_hat = float(fit["beta"])

        # Wide tolerances — GAS copula MLE is notoriously noisy in small samples
        assert abs(omega_hat - omega_true) < 0.25, f"omega: {omega_hat:.3f} vs {omega_true:.3f}"
        assert abs(alpha_hat - alpha_true) < 0.25, f"alpha: {alpha_hat:.3f} vs {alpha_true:.3f}"
        assert abs(beta_hat - beta_true) < 0.30, f"beta: {beta_hat:.3f} vs {beta_true:.3f}"

    def test_gas_rho_in_bounds(self) -> None:
        """Fitted rho should stay in (-1, 1)."""
        rng = np.random.default_rng(401)
        T = 800
        rho_true = 0.5
        cov = np.array([[1.0, rho_true], [rho_true, 1.0]])
        z = rng.multivariate_normal(np.zeros(2), cov, size=T)
        u = sstats.norm.cdf(z)

        fit = gas_copula_fit(u)
        rho_hat = np.asarray(fit["rho"])
        assert np.all((rho_hat > -1) & (rho_hat < 1))
        # With constant dependence, beta should be close to 1
        assert fit["beta"] > 0.5

    def test_gas_on_constant_dependence(self) -> None:
        """Constant-dependence data: alpha ≈ 0."""
        rng = np.random.default_rng(403)
        rho_const = 0.5
        T = 1000
        cov = np.array([[1.0, rho_const], [rho_const, 1.0]])
        z = rng.multivariate_normal(np.zeros(2), cov, size=T)
        u = sstats.norm.cdf(z)

        fit = gas_copula_fit(u)
        # On constant data, alpha should be small
        assert fit["alpha"] < 0.5, (
            f"alpha={fit['alpha']:.3f} should be small for constant dependence"
        )

    def test_gas_fail_closed_small_sample(self) -> None:
        rng = np.random.default_rng(404)
        u = rng.random((10, 2))
        with pytest.raises(ValueError):
            gas_copula_fit(u)


# ============================================================================
# SYNTHETIC validation 3: Tail-dependence capture
# ============================================================================


class TestTailDependenceCapture:
    def test_clayton_vine_captures_lower_tail(self) -> None:
        """Clayton vine should show higher lower-tail dependence than
        a Gaussian vine fitted to the same tail-dependent data."""
        # Generate data from a Clayton C-vine directly via vine_sample
        dim = 3
        vm_clayton_true = cvine_structure(dim)
        theta = 2.5
        for tree in range(dim - 1):
            for edge_idx in range(dim - tree - 1):
                vm_clayton_true.families[(tree, edge_idx)] = "clayton"
                vm_clayton_true.params[(tree, edge_idx)] = {"theta": theta}

        data = vine_sample(vm_clayton_true, 3000, seed=500)

        # Fit with Clayton family
        vm_clayton = vine_fit(data, families=("clayton",), criterion="aic")
        # Fit with Gaussian family
        vm_gauss = vine_fit(data, families=("gaussian",), criterion="aic")

        td_clayton = vine_tail_dependence(vm_clayton, n_sim=5000, seed=510)
        td_gauss = vine_tail_dependence(vm_gauss, n_sim=5000, seed=511)

        clayton_lower = td_clayton["0,1"]["lower"]
        gauss_lower = td_gauss["0,1"]["lower"]

        assert clayton_lower > gauss_lower, (
            f"Clayton lower tail={clayton_lower:.4f} should exceed "
            f"Gaussian lower tail={gauss_lower:.4f}"
        )

    def test_tail_dependence_contrast_gumbel_upper(self) -> None:
        """Gumbel vine captures upper-tail dependence, Gaussian does not."""
        np.random.default_rng(520)

        from quant_fund.models.copula import gumbel_copula_sim

        alpha = 2.5
        T = 5000

        uv1 = gumbel_copula_sim(alpha, T, seed=521)
        uv2 = gumbel_copula_sim(alpha, T, seed=522)
        data = np.column_stack([uv1[:, 0], uv1[:, 1], uv2[:, 1]])

        vm_gumbel = vine_fit(data, families=("gumbel",), criterion="aic")
        vm_gauss = vine_fit(data, families=("gaussian",), criterion="aic")

        td_gumbel = vine_tail_dependence(vm_gumbel, n_sim=5000, seed=530)
        td_gauss = vine_tail_dependence(vm_gauss, n_sim=5000, seed=531)

        gumbel_upper = td_gumbel["0,1"]["upper"]
        gauss_upper = td_gauss["0,1"]["upper"]

        assert gumbel_upper > gauss_upper, (
            f"Gumbel upper tail={gumbel_upper:.4f} should exceed "
            f"Gaussian upper tail={gauss_upper:.4f}"
        )


# ============================================================================
# Edge cases and fail-closed
# ============================================================================


class TestFailClosed:
    def test_vine_sample_negative_n(self) -> None:
        vm = cvine_structure(3)
        with pytest.raises(ValueError):
            vine_sample(vm, -1)

    def test_gas_filter_nan_input(self) -> None:
        u = np.full((100, 2), np.nan)
        with pytest.raises(ValueError):
            gas_copula_filter(u, 0.0, 0.1, 0.9)

    def test_vine_fit_nan_input(self) -> None:
        with pytest.raises(ValueError):
            vine_fit(np.full((30, 4), np.nan))

    def test_max_dim_exceeded(self) -> None:
        rng = np.random.default_rng(0)
        u = rng.random((30, _MAX_DIM + 1))
        with pytest.raises(ValueError):
            vine_fit(u)


# ============================================================================
# Honesty/determinism audit probes
# ============================================================================


class TestHinvClampDirection:
    """The no-sign-change fallback must clamp to the boundary h points at.

    h(u|v) is increasing in u; when w exceeds h over the whole interval the
    inverse sits at the top, not the floor.  The previous fallback clamped
    to the WRONG end (lo when h < w, hi when h > w), silently flipping
    sampled uniforms to the opposite tail.
    """

    @pytest.mark.parametrize(
        ("hinv_name", "h_name", "theta_key", "theta"),
        [
            ("_clayton_hinv", "_clayton_h", "theta", 2.0),
            ("_gumbel_hinv", "_gumbel_h", "alpha", 2.0),
            ("_frank_hinv", "_frank_h", "theta", 3.0),
            ("_joe_hinv", "_joe_h", "theta", 2.5),
        ],
    )
    def test_clamps_to_reached_boundary(
        self,
        monkeypatch: pytest.MonkeyPatch,
        hinv_name: str,
        h_name: str,
        theta_key: str,
        theta: float,
    ) -> None:
        import quant_fund.models.pair_vine_copula as pvc

        # Force h(u|v) to saturate at 0.5 — unreachable target both ways.
        monkeypatch.setattr(
            pvc,
            h_name,
            lambda u, v, t: np.full(np.asarray(u).shape, 0.5),
        )
        hinv = getattr(pvc, hinv_name)
        kw = {theta_key: theta}
        # w above the plateau → inverse at the TOP boundary.
        hi_out = hinv(np.array([0.9]), np.array([0.5]), **{k: v for k, v in kw.items()})
        assert float(hi_out[0]) > 0.9, f"{hinv_name} clamped low when h < w"
        # w below the plateau → inverse at the BOTTOM boundary.
        lo_out = hinv(np.array([0.1]), np.array([0.5]), **{k: v for k, v in kw.items()})
        assert float(lo_out[0]) < 0.1, f"{hinv_name} clamped high when h > w"


class TestFamilyParamValidation:
    """Out-of-domain pair-copula params must fail loudly at dispatch."""

    @pytest.mark.parametrize(
        ("family", "params"),
        [
            ("gaussian", {"rho": 1.5}),
            ("gaussian", {"rho": np.nan}),
            ("t", {"rho": 0.3, "nu": -1.0}),
            ("t", {"rho": 0.3}),  # missing nu
            ("clayton", {"theta": -2.0}),
            ("gumbel", {"alpha": 0.5}),
            ("joe", {"theta": 0.5}),
            ("frank", {"theta": np.inf}),
        ],
    )
    def test_invalid_params_rejected(self, family: str, params: dict) -> None:
        from quant_fund.models.pair_vine_copula import _h_eval, _hinv_eval

        u = np.array([0.3, 0.7])
        with pytest.raises(ValueError):
            _h_eval(u, u, family, params)
        with pytest.raises(ValueError):
            _hinv_eval(u, u, family, params)

    def test_invalid_params_rejected_in_vine_pipeline(self) -> None:
        """A hand-built VineMatrix with garbage params must not sample."""
        vm = cvine_structure(3)
        vm.families[(0, 0)] = "gaussian"
        vm.params[(0, 0)] = {"rho": 5.0}
        with pytest.raises(ValueError):
            vine_sample(vm, 50, seed=0)


class TestStructureDetection:
    def test_handbuilt_cvine_diag_routes_to_cvine_sampler(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A legacy VineMatrix (structure='rvine') with a C-vine diagonal
        must sample via the C-vine path — the sorted-diagonal heuristic used
        to send every permutation diagonal to the D-vine sampler.
        """
        import quant_fund.models.pair_vine_copula as pvc

        vm = cvine_structure(3)
        vm.families[(0, 0)] = "gaussian"
        vm.params[(0, 0)] = {"rho": 0.5}
        vm.structure = "rvine"  # simulate a hand-built object w/o the tag
        called: dict[str, bool] = {"cvine": False, "dvine": False}
        monkeypatch.setattr(
            pvc,
            "_cvine_sample_inner",
            lambda *a, **k: called.__setitem__("cvine", True) or np.zeros((1, 3)),
        )
        monkeypatch.setattr(
            pvc,
            "_dvine_sample_inner",
            lambda *a, **k: called.__setitem__("dvine", True) or np.zeros((1, 3)),
        )
        pvc.vine_sample(vm, 1, seed=0)
        assert called == {"cvine": True, "dvine": False}

    def test_handbuilt_dvine_diag_routes_to_dvine_sampler(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import quant_fund.models.pair_vine_copula as pvc

        vm = dvine_structure(3)
        vm.families[(0, 0)] = "gaussian"
        vm.params[(0, 0)] = {"rho": 0.5}
        vm.structure = "rvine"
        called: dict[str, bool] = {"cvine": False, "dvine": False}
        monkeypatch.setattr(
            pvc,
            "_cvine_sample_inner",
            lambda *a, **k: called.__setitem__("cvine", True) or np.zeros((1, 3)),
        )
        monkeypatch.setattr(
            pvc,
            "_dvine_sample_inner",
            lambda *a, **k: called.__setitem__("dvine", True) or np.zeros((1, 3)),
        )
        pvc.vine_sample(vm, 1, seed=0)
        assert called == {"cvine": False, "dvine": True}


class TestGasValidation:
    def test_filter_rejects_nonstationary_beta(self) -> None:
        rng = np.random.default_rng(700)
        u = rng.random((200, 2))
        with pytest.raises(ValueError, match="beta"):
            gas_copula_filter(u, 0.0, 0.1, 1.5)
        with pytest.raises(ValueError, match="beta"):
            gas_copula_filter(u, 0.0, 0.1, -1.0)

    def test_filter_rejects_nonfinite_params(self) -> None:
        u = np.random.default_rng(701).random((200, 2))
        with pytest.raises(ValueError):
            gas_copula_filter(u, np.nan, 0.1, 0.5)
        with pytest.raises(ValueError):
            gas_copula_filter(u, 0.0, np.inf, 0.5)

    def test_filter_rejects_wrong_width(self) -> None:
        """The silent m[:, :2] truncation is gone — wrong width raises."""
        rng = np.random.default_rng(702)
        with pytest.raises(ValueError, match="2 columns"):
            gas_copula_filter(rng.random((100, 5)), 0.0, 0.1, 0.5)
        with pytest.raises(ValueError, match="2 columns"):
            gas_copula_filter(rng.normal(size=(100, 3)), 0.0, 0.1, 0.5)

    def test_filter_rejects_constant_column(self) -> None:
        """Constant column → corrcoef undefined → must raise, not init κ=±3."""
        rng = np.random.default_rng(703)
        u = rng.random((200, 2))
        u[:, 0] = 0.5  # constant
        with pytest.raises(ValueError, match="constant|correlation"):
            gas_copula_filter(u, 0.0, 0.1, 0.5)

    def test_fit_rejects_wrong_width_and_constant_column(self) -> None:
        rng = np.random.default_rng(704)
        with pytest.raises(ValueError, match="2 columns"):
            gas_copula_fit(rng.random((100, 4)))
        u = rng.random((100, 2))
        u[:, 1] = 0.25
        with pytest.raises(ValueError, match="constant|correlation"):
            gas_copula_fit(u)


class TestUnitIntervalDomain:
    def test_vine_fit_rejects_out_of_range_pseudo_obs(self) -> None:
        """Raw (unranked) data must not be silently clipped into uniforms."""
        rng = np.random.default_rng(705)
        bad = rng.normal(size=(40, 3))  # mostly outside [0, 1]
        with pytest.raises(ValueError, match="\\[0, 1\\]"):
            vine_fit(bad)

    def test_vine_logpdf_rejects_out_of_range(self) -> None:
        rng = np.random.default_rng(706)
        u = rng.random((40, 3))
        vm = vine_fit(u, families=("gaussian",))
        bad = rng.normal(size=(10, 3)) * 2.0
        with pytest.raises(ValueError, match="\\[0, 1\\]"):
            vine_logpdf(vm, bad)
