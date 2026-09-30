"""Tests for models/dynamic_subspace_denoising.py (Wouters & Diks 2026).

All data is seeded SYNTHETIC planted structure: a d-dimensional AR(1) latent
dynamic component observed in n dimensions with additive white noise.  These
are correctness tests only — never market evidence.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from quant_fund.models.dynamic_subspace_denoising import (
    bootstrap_dimension_select,
    dynamic_space_matrix,
    estimate_dynamic_subspace,
    lagged_autocovariance,
    optimal_projection_denoise,
    orthogonal_projection_denoise,
    principal_angles,
)

SEED = 20260929
N_DIM = 12
D_TRUE = 3
T_LEN = 600


def _orthonormal_u(rng: np.random.Generator, n: int, d: int) -> np.ndarray:
    g = rng.standard_normal((n, d))
    q, _ = np.linalg.qr(g)
    return q


def _plant_ar1_panel(
    seed: int,
    *,
    n: int = N_DIM,
    d: int = D_TRUE,
    t_len: int = T_LEN,
    noise_scale: float = 0.5,
    oblique: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (y, x_true, U): observed panel, latent signal, dynamic basis."""
    rng = np.random.default_rng(seed)
    u = _orthonormal_u(rng, n, d)
    phi = np.diag([0.9, 0.7, 0.5][:d] + [0.3] * max(0, d - 3))
    xi = np.zeros((t_len, d))
    eta = rng.standard_normal((t_len, d))
    for i in range(1, t_len):
        xi[i] = phi @ xi[i - 1] + eta[i]
    x = xi @ u.T  # (T, n)

    if not oblique:
        eps = noise_scale * rng.standard_normal((t_len, n))
    else:
        # Structured noise: one noise direction is oblique to the dynamic
        # space, creating Cov[eps_par, eps_perp] != 0 (the regime where the
        # optimal oblique projection strictly beats the orthogonal one).
        q_perp = _orthonormal_u(rng, n, n - d)[:, :2]
        s1 = u[:, 0] + q_perp[:, 0]
        s1 /= np.linalg.norm(s1)
        s2 = u[:, 1] + q_perp[:, 1]
        s2 /= np.linalg.norm(s2)
        s = np.column_stack([s1, s2])
        w = rng.standard_normal((t_len, 2))
        eps = noise_scale * (w @ s.T) + 0.1 * noise_scale * rng.standard_normal((t_len, n))

    y = x + eps
    return y, x, u


# ---------------------------------------------------------------------------
# lagged_autocovariance
# ---------------------------------------------------------------------------


class TestLaggedAutocovariance:
    def test_k0_symmetric(self) -> None:
        y, _, _ = _plant_ar1_panel(SEED)
        cov0 = lagged_autocovariance(y, 0)
        np.testing.assert_allclose(cov0, cov0.T, atol=1e-12)

    def test_shape(self) -> None:
        y, _, _ = _plant_ar1_panel(SEED)
        cov1 = lagged_autocovariance(y, 1)
        assert cov1.shape == (N_DIM, N_DIM)

    def test_lagged_covariance_tracks_signal(self) -> None:
        """At k != 0 the naive lagged covariance is unbiased for the SIGNAL
        autocovariance (white noise vanishes at nonzero lags)."""
        y, x, _ = _plant_ar1_panel(SEED)
        cov_y = lagged_autocovariance(y, 3)
        cov_x = lagged_autocovariance(x, 3)
        # Estimate on a long panel should be close (same seed, T=600)
        rel = np.linalg.norm(cov_y - cov_x) / max(np.linalg.norm(cov_x), 1e-12)
        assert rel < 0.35

    def test_fail_closed(self) -> None:
        y, _, _ = _plant_ar1_panel(SEED)
        with pytest.raises(ValueError):
            lagged_autocovariance(y, -1)
        with pytest.raises(ValueError):
            lagged_autocovariance(y, 10_000)
        with pytest.raises(ValueError):
            lagged_autocovariance(np.ones((10, 3)), 1)  # too few rows
        with pytest.raises(ValueError):
            lagged_autocovariance(np.full((30, 3), np.nan), 1)
        with pytest.raises(ValueError):
            lagged_autocovariance(np.ones(30), 1)  # 1-D


# ---------------------------------------------------------------------------
# dynamic_space_matrix
# ---------------------------------------------------------------------------


class TestDynamicSpaceMatrix:
    def test_symmetric_psd(self) -> None:
        y, _, _ = _plant_ar1_panel(SEED)
        k = dynamic_space_matrix(y)
        np.testing.assert_allclose(k, k.T, atol=1e-10)
        vals = np.linalg.eigvalsh(k)
        assert vals.min() > -1e-10

    def test_top_d_eigenvalues_dominate(self) -> None:
        """K has exactly d nonzero population eigenvalues; sample spectrum
        should show a large gap after the d-th."""
        y, _, _ = _plant_ar1_panel(SEED)
        k = dynamic_space_matrix(y)
        vals = np.linalg.eigvalsh(k)[::-1]
        gap = vals[D_TRUE - 1] / max(vals[D_TRUE], 1e-15)
        assert gap > 10.0

    def test_fail_closed(self) -> None:
        y, _, _ = _plant_ar1_panel(SEED)
        with pytest.raises(ValueError):
            dynamic_space_matrix(y, k0=0)
        with pytest.raises(ValueError):
            dynamic_space_matrix(y, k0=2, coefficients=[1.0, -1.0])
        with pytest.raises(ValueError):
            dynamic_space_matrix(y, k0=2, coefficients=[1.0])


# ---------------------------------------------------------------------------
# estimate_dynamic_subspace
# ---------------------------------------------------------------------------


class TestEstimateDynamicSubspace:
    def test_oracle_basis_recovers_dynamic_space(self) -> None:
        y, _, u = _plant_ar1_panel(SEED)
        sub = estimate_dynamic_subspace(y, d=D_TRUE)
        assert sub["d"] == D_TRUE
        assert sub["bootstrapped"] is False
        angles = principal_angles(sub["basis"], u)
        # All principal angles small (subspaces align)
        assert float(np.max(angles)) < 0.35

    def test_eigenvalues_descending(self) -> None:
        y, _, _ = _plant_ar1_panel(SEED)
        sub = estimate_dynamic_subspace(y, d=D_TRUE)
        vals = sub["eigenvalues"]
        assert np.all(np.diff(vals) <= 1e-12)

    def test_bootstrap_requires_seed(self) -> None:
        y, _, _ = _plant_ar1_panel(SEED)
        with pytest.raises(ValueError):
            estimate_dynamic_subspace(y, d=None)

    def test_fail_closed_dim(self) -> None:
        y, _, _ = _plant_ar1_panel(SEED)
        with pytest.raises(ValueError):
            estimate_dynamic_subspace(y, d=0)
        with pytest.raises(ValueError):
            estimate_dynamic_subspace(y, d=N_DIM)  # must be <= n-1


# ---------------------------------------------------------------------------
# bootstrap_dimension_select
# ---------------------------------------------------------------------------


class TestBootstrapDimensionSelect:
    def test_recovers_true_dimension(self) -> None:
        y, _, _ = _plant_ar1_panel(SEED, t_len=800)
        res = bootstrap_dimension_select(y, seed=SEED, n_boot=200)
        # Exact recovery or within one of the truth (sequential tests can
        # stop one step early under MC noise)
        assert abs(res["d"] - D_TRUE) <= 1

    def test_determinism(self) -> None:
        y, _, _ = _plant_ar1_panel(SEED)
        r1 = bootstrap_dimension_select(y, seed=7, n_boot=100)
        r2 = bootstrap_dimension_select(y, seed=7, n_boot=100)
        assert r1["d"] == r2["d"]

    def test_pure_noise_gives_small_d(self) -> None:
        """A white-noise panel has no dynamic space; the sequential test
        should stop at a small dimension."""
        rng = np.random.default_rng(SEED)
        y = rng.standard_normal((600, N_DIM))
        res = bootstrap_dimension_select(y, seed=SEED, n_boot=150)
        assert res["d"] <= 2

    def test_fail_closed(self) -> None:
        y, _, _ = _plant_ar1_panel(SEED)
        with pytest.raises(ValueError):
            bootstrap_dimension_select(y, seed=SEED, alpha=0.0)
        with pytest.raises(ValueError):
            bootstrap_dimension_select(y, seed=SEED, alpha=1.0)


# ---------------------------------------------------------------------------
# Denoising: optimal vs orthogonal vs raw
# ---------------------------------------------------------------------------


class TestDenoising:
    def test_isotropic_noise_optimal_near_orthogonal(self) -> None:
        """With iid isotropic noise, Cov[eps_par, eps_perp] = 0 and the
        optimal projection approaches the orthogonal one (module docstring;
        finite-sample estimation noise keeps them from being identical).
        Contrast: under structured oblique noise the gap is much larger."""
        y, _, _ = _plant_ar1_panel(SEED, oblique=False, t_len=1200)
        opt = optimal_projection_denoise(y, d=D_TRUE)
        ortho = orthogonal_projection_denoise(y, d=D_TRUE)
        iso_gap = float(np.max(np.abs(opt["projection"] - ortho["projection"])))
        assert iso_gap < 0.15

        y_ob, _, _ = _plant_ar1_panel(SEED, oblique=True, t_len=1200)
        opt_ob = optimal_projection_denoise(y_ob, d=D_TRUE)
        ortho_ob = orthogonal_projection_denoise(y_ob, d=D_TRUE)
        oblique_gap = float(np.max(np.abs(opt_ob["projection"] - ortho_ob["projection"])))
        assert oblique_gap > iso_gap

    def test_oblique_noise_optimal_strictly_beats_orthogonal(self) -> None:
        """With structured (oblique) noise the optimal projection removes the
        eps_par component predictable from eps_perp: MSE ordering
        optimal < orthogonal < raw (paper Theorem 1/2)."""
        y, x, _ = _plant_ar1_panel(SEED, oblique=True, t_len=1500)
        opt = optimal_projection_denoise(y, d=D_TRUE)
        ortho = orthogonal_projection_denoise(y, d=D_TRUE)
        mse_raw = float(np.mean((y - x) ** 2))
        mse_ortho = float(np.mean((ortho["denoised"] - x) ** 2))
        mse_opt = float(np.mean((opt["denoised"] - x) ** 2))
        assert mse_opt < mse_ortho < mse_raw

    def test_optimal_projection_oblique_under_structured_noise(self) -> None:
        """P_opt is generally non-symmetric (oblique) when noise is
        structured."""
        y, _, _ = _plant_ar1_panel(SEED, oblique=True, t_len=1500)
        opt = optimal_projection_denoise(y, d=D_TRUE)
        p = opt["projection"]
        asym = float(np.max(np.abs(p - p.T)))
        assert asym > 1e-8

    def test_orthogonal_projection_symmetric_idempotent(self) -> None:
        y, _, _ = _plant_ar1_panel(SEED)
        ortho = orthogonal_projection_denoise(y, d=D_TRUE)
        p = ortho["projection"]
        np.testing.assert_allclose(p, p.T, atol=1e-10)
        np.testing.assert_allclose(p @ p, p, atol=1e-8)

    def test_denoised_shape(self) -> None:
        y, _, _ = _plant_ar1_panel(SEED)
        opt = optimal_projection_denoise(y, d=D_TRUE)
        assert opt["denoised"].shape == y.shape

    def test_mse_ordering_seed_robust(self) -> None:
        """The MSE ordering optimal < orthogonal < raw holds across seeds
        under structured oblique noise (the module's central claim)."""
        for seed in (SEED, SEED + 1, SEED + 2):
            y, x, _ = _plant_ar1_panel(seed, oblique=True, t_len=1200)
            opt = optimal_projection_denoise(y, d=D_TRUE)
            ortho = orthogonal_projection_denoise(y, d=D_TRUE)
            mse_raw = float(np.mean((y - x) ** 2))
            mse_ortho = float(np.mean((ortho["denoised"] - x) ** 2))
            mse_opt = float(np.mean((opt["denoised"] - x) ** 2))
            assert mse_opt < mse_ortho < mse_raw, f"ordering failed at seed {seed}"

    def test_determinism(self) -> None:
        y, _, _ = _plant_ar1_panel(SEED)
        r1 = optimal_projection_denoise(y, d=D_TRUE)
        r2 = optimal_projection_denoise(y, d=D_TRUE)
        np.testing.assert_array_equal(r1["projection"], r2["projection"])

    def test_fail_closed(self) -> None:
        y, _, _ = _plant_ar1_panel(SEED)
        with pytest.raises(ValueError):
            optimal_projection_denoise(y, d=D_TRUE, tau_perp=0.0)
        with pytest.raises(ValueError):
            optimal_projection_denoise(y, d=D_TRUE, tau_perp=1.0)


# ---------------------------------------------------------------------------
# Parametric-rate illustration
# ---------------------------------------------------------------------------


class TestParametricRate:
    def test_error_shrinks_with_T(self) -> None:
        """Estimation error of the denoiser shrinks with T — illustration of
        the O_P(T^{-1/2}) rate (Theorem 3); monotone check on a T grid."""
        mses = []
        for t_len in (300, 1200):
            y, x, _ = _plant_ar1_panel(SEED + 1, oblique=True, t_len=t_len)
            opt = optimal_projection_denoise(y, d=D_TRUE)
            mses.append(float(np.mean((opt["denoised"] - x) ** 2)))
        assert mses[1] < mses[0]


# ---------------------------------------------------------------------------
# principal_angles
# ---------------------------------------------------------------------------


class TestPrincipalAngles:
    def test_identical_subspace_zero(self) -> None:
        rng = np.random.default_rng(SEED)
        q, _ = np.linalg.qr(rng.standard_normal((10, 3)))
        angles = principal_angles(q, q)
        np.testing.assert_allclose(angles, 0.0, atol=1e-8)

    def test_orthogonal_subspace_half_pi(self) -> None:
        a = np.eye(6)[:, :2]
        b = np.eye(6)[:, 2:4]
        angles = principal_angles(a, b)
        np.testing.assert_allclose(angles, math.pi / 2, atol=1e-8)

    def test_angles_ascending(self) -> None:
        rng = np.random.default_rng(SEED)
        qa, _ = np.linalg.qr(rng.standard_normal((10, 3)))
        qb, _ = np.linalg.qr(rng.standard_normal((10, 4)))
        angles = principal_angles(qa, qb)
        assert np.all(np.diff(angles) >= -1e-12)
        assert angles.shape == (3,)

    def test_fail_closed_not_orthonormal(self) -> None:
        a = np.ones((5, 2))
        b = np.eye(5)[:, :2]
        with pytest.raises(ValueError):
            principal_angles(a, b)

    def test_fail_closed_shape(self) -> None:
        b = np.eye(5)[:, :2]
        with pytest.raises(ValueError):
            principal_angles(np.eye(4)[:, :2], b)
        with pytest.raises(ValueError):
            principal_angles(np.eye(5)[:, :0], b)
