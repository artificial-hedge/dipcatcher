"""Unit tests for metrics.marchenko_pastur — SYNTHETIC correctness only."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.metrics.marchenko_pastur import (
    bench_marchenko_pastur,
    eigenvalue_clip,
    mp_bounds,
    mp_density,
    mp_edge_test,
    rotation_shrink,
    sim_correlated,
)

FORBIDDEN_TOKENS = frozenset({"sharpe", "sortino", "calmar", "pnl", "nav"})


def _sample_cov(x: np.ndarray) -> np.ndarray:
    return np.asarray(x.T @ x / x.shape[0], dtype=np.float64)


def _eigvals(m: np.ndarray) -> np.ndarray:
    return np.asarray(np.linalg.eigvalsh(m), dtype=np.float64)


# ------------------------------------------------------------- mp_density --


def test_density_normalizes_to_one() -> None:
    for q in (0.25, 0.5, 0.9):
        lo, hi = mp_bounds(q)
        grid = np.linspace(1e-9, hi, 20001)
        mass = float(np.trapezoid(mp_density(grid, q), grid))
        assert mass == pytest.approx(1.0, abs=2e-4)


def test_density_support_zero_outside() -> None:
    q = 0.4
    lo, hi = mp_bounds(q)
    below = np.linspace(0.0, lo - 1e-6, 50)
    above = np.linspace(hi + 1e-6, 4.0 * hi, 50)
    assert np.all(mp_density(below, q) == 0.0)
    assert np.all(mp_density(above, q) == 0.0)
    inside = np.linspace(lo + 1e-6, hi - 1e-6, 50)
    assert np.all(mp_density(inside, q) > 0.0)


def test_density_endpoints_vanish_inside() -> None:
    q = 0.3
    lo, hi = mp_bounds(q)
    # just inside the support the density is finite and small near edges
    eps = 1e-9
    d_lo = float(mp_density(np.array([lo + eps]), q)[0])
    d_hi = float(mp_density(np.array([hi - eps]), q)[0])
    d_mid = float(mp_density(np.array([(lo + hi) / 2.0]), q)[0])
    assert d_lo < d_mid and d_hi < d_mid


def test_density_sigma2_scaling() -> None:
    q = 0.5
    x = np.linspace(0.1, 3.0, 25)
    s2 = 2.5
    a = mp_density(x * s2, q, sigma2=s2) * s2
    b = mp_density(x, q, sigma2=1.0)
    assert np.allclose(a, b, rtol=1e-10)


def test_density_q_gt_one_defective_mass() -> None:
    q = 2.0
    lo, hi = mp_bounds(q)
    grid = np.linspace(lo + 1e-9, hi, 20001)
    mass = float(np.trapezoid(mp_density(grid, q), grid))
    # continuous part carries mass 1/q; the rest is the atom at zero
    assert mass == pytest.approx(1.0 / q, abs=2e-3)


def test_density_q1_diverges_at_zero() -> None:
    out = mp_density(np.array([0.0, 1e-6]), 1.0)
    assert out[0] == np.inf
    assert np.isfinite(out[1]) and out[1] > 0.0


def test_density_rejects_bad_params() -> None:
    x = np.array([1.0])
    for bad_q in (0.0, -0.5, np.inf, np.nan):
        with pytest.raises(ValueError):
            mp_density(x, bad_q)
    for bad_s2 in (0.0, -1.0, np.inf, np.nan):
        with pytest.raises(ValueError):
            mp_density(x, 0.5, sigma2=bad_s2)
    with pytest.raises(ValueError):
        mp_density(np.array([1.0, np.nan]), 0.5)


# -------------------------------------------------------------- mp_bounds --


def test_bounds_known_values() -> None:
    lo, hi = mp_bounds(0.25, 1.0)
    assert lo == pytest.approx(0.25)
    assert hi == pytest.approx(2.25)


def test_bounds_q1_and_sigma2() -> None:
    lo, hi = mp_bounds(1.0, 2.0)
    assert lo == pytest.approx(0.0)
    assert hi == pytest.approx(8.0)


def test_bounds_scale_with_sigma2() -> None:
    lo1, hi1 = mp_bounds(0.4, 1.0)
    lo2, hi2 = mp_bounds(0.4, 3.0)
    assert lo2 == pytest.approx(3.0 * lo1)
    assert hi2 == pytest.approx(3.0 * hi1)


def test_bounds_reject_bad_params() -> None:
    for bad in (0.0, -1.0, np.inf, np.nan):
        with pytest.raises(ValueError):
            mp_bounds(bad)
    with pytest.raises(ValueError):
        mp_bounds(0.5, sigma2=0.0)


# ------------------------------------------------------------ mp_edge_test --


def test_edge_test_detects_planted_factors() -> None:
    n_obs, p, k = 500, 50, 4
    x = sim_correlated(n_obs, p, k, 0.5, seed=7)
    ev = _eigvals(_sample_cov(x))
    out = mp_edge_test(ev, p / n_obs, n_obs)
    assert out["count_above_edge"] == pytest.approx(k, abs=1.0)
    assert out["share_above_tw_edge"] * p <= k
    assert out["edge"] > 0.0
    assert out["edge_tw"] > out["edge"] - 1e-9  # soft edge sits near/above hard edge
    assert 0.0 <= out["share_above_edge"] <= 1.0


def test_edge_test_pure_noise_small_share() -> None:
    n_obs, p = 400, 80
    x = sim_correlated(n_obs, p, 0, 1.0, seed=11)
    ev = _eigvals(_sample_cov(x))
    out = mp_edge_test(ev, p / n_obs, n_obs, sigma2=1.0)
    assert out["share_above_edge"] < 0.10
    assert out["share_above_tw_edge"] <= out["share_above_edge"]
    assert out["edge_tw"] > out["edge"]  # soft edge sits above the hard edge


def test_edge_test_tw_band_widens_detection_threshold() -> None:
    # spectrum consistent with q = p/n (50 eigenvalues, q=0.2, n=250)
    rng = np.random.default_rng(3)
    x = rng.standard_normal((250, 50))
    ev = _eigvals(_sample_cov(x))
    out = mp_edge_test(ev, 0.2, 250, sigma2=1.0)
    assert out["edge_tw"] > out["edge"]
    assert out["share_above_tw_edge"] <= out["share_above_edge"]


def test_edge_test_rejects_bad_inputs() -> None:
    ev = np.array([0.5, 1.0])
    with pytest.raises(ValueError):
        mp_edge_test(ev, 0.0, 100)
    with pytest.raises(ValueError):
        mp_edge_test(ev, 0.5, 1)
    with pytest.raises(ValueError):
        mp_edge_test(np.array([]), 0.5, 100)
    with pytest.raises(ValueError):
        mp_edge_test(np.array([[1.0, 2.0]]), 0.5, 100)
    with pytest.raises(ValueError):
        mp_edge_test(np.array([1.0, np.inf]), 0.5, 100)
    with pytest.raises(ValueError):
        mp_edge_test(ev, 0.5, 100, sigma2=-1.0)
    with pytest.raises(ValueError):
        mp_edge_test(ev, 0.5, 100, tw_quantile=np.nan)
    with pytest.raises(ValueError):
        mp_edge_test(np.array([-1.0, -2.0]), 0.5, 100)  # no positive median


# --------------------------------------------------------- eigenvalue_clip --


def test_clip_preserves_trace() -> None:
    x = sim_correlated(300, 40, 3, 0.5, seed=3)
    cov = _sample_cov(x)
    clipped = eigenvalue_clip(cov, 300)
    assert float(np.trace(clipped)) == pytest.approx(float(np.trace(cov)), rel=1e-10)


def test_clip_replaces_bulk_with_bulk_mean() -> None:
    x = sim_correlated(400, 30, 2, 0.7, seed=5)
    cov = _sample_cov(x)
    clipped = eigenvalue_clip(cov, 400)
    e_new = _eigvals(clipped)
    e_old = _eigvals(cov)
    # bulk eigenvalues collapse onto a single value
    vals, counts = np.unique(np.round(e_new, 8), return_counts=True)
    top = vals[np.argmax(counts)]
    n_bulk = int(np.sum(e_old <= e_new[np.argmax(counts)] + 1e-6))
    assert counts.max() >= n_bulk - 5
    assert e_new[-1] >= top  # signal survives at the top


def test_clip_improves_frobenius_on_factor_data() -> None:
    n_obs, p, k, noise = 300, 50, 4, 0.5
    rng = np.random.default_rng(9)
    b = rng.standard_normal((p, k))
    s = np.sqrt((1.0 - noise**2) / k)
    sigma_true = (b * s) @ (b * s).T + noise**2 * np.eye(p)
    f = rng.standard_normal((n_obs, k))
    z = rng.standard_normal((n_obs, p))
    x = f @ (b * s).T + noise * z
    cov = _sample_cov(x)
    clipped = eigenvalue_clip(cov, n_obs)
    err_raw = float(np.linalg.norm(cov - sigma_true, "fro"))
    err_clip = float(np.linalg.norm(clipped - sigma_true, "fro"))
    assert err_clip < err_raw


def test_clip_output_symmetric_psd() -> None:
    x = sim_correlated(200, 25, 2, 0.6, seed=13)
    clipped = eigenvalue_clip(_sample_cov(x), 200)
    assert np.allclose(clipped, clipped.T, atol=1e-12)
    assert float(np.min(np.linalg.eigvalsh(clipped))) >= -1e-10


def test_clip_pure_noise_flattens_toward_bulk() -> None:
    n_obs, p = 500, 50
    x = sim_correlated(n_obs, p, 0, 1.0, seed=17)
    cov = _sample_cov(x)
    clipped = eigenvalue_clip(cov, n_obs)
    spread_raw = float(np.max(_eigvals(cov)) - np.min(_eigvals(cov)))
    spread_clip = float(np.max(_eigvals(clipped)) - np.min(_eigvals(clipped)))
    assert spread_clip < spread_raw


def test_clip_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError):
        eigenvalue_clip(np.ones((3, 4)), 100)
    with pytest.raises(ValueError):
        eigenvalue_clip(np.array([[1.0, np.nan], [0.0, 1.0]]), 100)
    with pytest.raises(ValueError):
        eigenvalue_clip(np.eye(4), 0)
    with pytest.raises(ValueError):
        eigenvalue_clip(np.eye(4), 2.5)  # type: ignore[arg-type]


# --------------------------------------------------------- rotation_shrink --


def test_shrink_alpha_zero_is_identity() -> None:
    x = sim_correlated(120, 20, 2, 0.5, seed=19)
    cov = _sample_cov(x)
    out = rotation_shrink(cov, 120, alpha=0.0)
    assert np.allclose(out, cov, atol=1e-12)


def test_shrink_alpha_one_is_scaled_identity() -> None:
    x = sim_correlated(120, 20, 2, 0.5, seed=21)
    cov = _sample_cov(x)
    out = rotation_shrink(cov, 120, alpha=1.0)
    mu = float(np.trace(cov)) / cov.shape[0]
    assert np.allclose(out, mu * np.eye(cov.shape[0]), atol=1e-10)


def test_shrink_oas_alpha_in_bounds_and_trace_preserving() -> None:
    n_obs = 250
    x = sim_correlated(n_obs, 40, 3, 0.5, seed=23)
    cov = _sample_cov(x)
    out = rotation_shrink(cov, n_obs)
    assert float(np.trace(out)) == pytest.approx(float(np.trace(cov)), rel=1e-10)
    # OAS shrinks the spectrum toward its mean: spread must not grow
    assert np.max(_eigvals(out)) <= np.max(_eigvals(cov)) + 1e-9
    assert np.min(_eigvals(out)) >= np.min(_eigvals(cov)) - 1e-9


def test_shrink_improves_frobenius_on_factor_data() -> None:
    n_obs, p, k, noise = 200, 60, 5, 0.6
    rng = np.random.default_rng(27)
    b = rng.standard_normal((p, k))
    s = np.sqrt((1.0 - noise**2) / k)
    sigma_true = (b * s) @ (b * s).T + noise**2 * np.eye(p)
    x = rng.standard_normal((n_obs, k)) @ (b * s).T + noise * rng.standard_normal((n_obs, p))
    cov = _sample_cov(x)
    shrunk = rotation_shrink(cov, n_obs)
    assert float(np.linalg.norm(shrunk - sigma_true, "fro")) < float(
        np.linalg.norm(cov - sigma_true, "fro")
    )


def test_shrink_flat_spectrum_gives_identity_target() -> None:
    cov = 1.5 * np.eye(6)
    out = rotation_shrink(cov, 100)
    assert np.allclose(out, cov, atol=1e-12)


def test_shrink_rejects_bad_inputs() -> None:
    cov = np.eye(3)
    for bad in (-0.1, 1.1, np.nan, np.inf):
        with pytest.raises(ValueError):
            rotation_shrink(cov, 100, alpha=bad)
    with pytest.raises(ValueError):
        rotation_shrink(np.ones((2, 3)), 100)
    with pytest.raises(ValueError):
        rotation_shrink(-np.eye(3), 100)  # non-positive trace


# --------------------------------------------------------- sim_correlated --


def test_sim_shape_dtype_determinism() -> None:
    x1 = sim_correlated(64, 12, 3, 0.4, seed=42)
    x2 = sim_correlated(64, 12, 3, 0.4, seed=42)
    x3 = sim_correlated(64, 12, 3, 0.4, seed=43)
    assert x1.shape == (64, 12) and x1.dtype == np.float64
    assert np.array_equal(x1, x2)
    assert not np.array_equal(x1, x3)


def test_sim_marginal_variance_near_one() -> None:
    x = sim_correlated(2000, 60, 4, 0.5, seed=31)
    var = np.var(x, axis=0)
    assert np.all(np.abs(var - 1.0) < 0.15)


def test_sim_pure_noise_spectrum_inside_mp_bulk() -> None:
    n_obs, p = 400, 60
    x = sim_correlated(n_obs, p, 0, 1.0, seed=33)
    ev = _eigvals(_sample_cov(x))
    lo, hi = mp_bounds(p / n_obs, 1.0)
    inside = (ev >= lo * 0.9) & (ev <= hi * 1.1)
    assert float(np.mean(inside.astype(float))) > 0.9


def test_sim_planted_factors_exceed_edge() -> None:
    n_obs, p, k = 500, 50, 3
    x = sim_correlated(n_obs, p, k, 0.5, seed=35)
    ev = _eigvals(_sample_cov(x))
    _, hi = mp_bounds(p / n_obs, 1.0)
    assert int(np.sum(ev > hi)) >= k


def test_sim_rejects_bad_params() -> None:
    with pytest.raises(ValueError):
        sim_correlated(1, 10, 1, 0.5, 0)
    with pytest.raises(ValueError):
        sim_correlated(10, 0, 1, 0.5, 0)
    with pytest.raises(ValueError):
        sim_correlated(10, 5, -1, 0.5, 0)
    with pytest.raises(ValueError):
        sim_correlated(10, 5, 1, 1.5, 0)
    with pytest.raises(ValueError):
        sim_correlated(10, 5, 1, -0.1, 0)
    with pytest.raises(ValueError):
        sim_correlated(10, 5, 1, 0.5, 1.5)  # type: ignore[arg-type]


# -------------------------------------------------------------- bench ------


def test_bench_keys_all_synthetic_floats() -> None:
    out = bench_marchenko_pastur(0)
    assert len(out) >= 10
    for key, val in out.items():
        assert key.startswith("synthetic_"), key
        assert isinstance(val, float), key
        assert np.isfinite(val), key
        tokens = set(key.lower().split("_"))
        assert not tokens & FORBIDDEN_TOKENS, key


def test_bench_deterministic() -> None:
    a = bench_marchenko_pastur(7)
    b = bench_marchenko_pastur(7)
    assert a == b


def test_bench_signal_detection_and_denoise() -> None:
    out = bench_marchenko_pastur(0)
    assert out["synthetic_signal_count_err"] <= 1.0
    assert out["synthetic_clip_fro_improvement"] > 0.0
    assert out["synthetic_shrink_fro_improvement"] > 0.0
    assert out["synthetic_clip_trace_err"] < 1e-9
    assert out["synthetic_shrink_trace_err"] < 1e-9
    assert out["synthetic_determinism"] == 1.0
    assert out["synthetic_density_mass_err"] < 1e-3
    assert 0.0 <= out["synthetic_oas_alpha"] <= 1.0
    assert out["synthetic_tw_edge_viol_rate"] < 0.1


def test_bench_rejects_bad_seed() -> None:
    with pytest.raises(ValueError):
        bench_marchenko_pastur(1.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        bench_marchenko_pastur(True)  # type: ignore[arg-type]
