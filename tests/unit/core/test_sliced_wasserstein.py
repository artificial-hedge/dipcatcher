"""Tests for sliced_wasserstein: SW/MSW distances, permutation test, barycenter.

Every random draw uses a seeded np.random.default_rng, so runs are
deterministic and pinned. Anchors: d=1 exactness against wasserstein_1d,
the SW <= W2 <= sqrt(d) SW sandwich on Gaussians (wasserstein_gaussian
closed form; Nadjahi et al. 2021), permutation-test size ~alpha under the
null and power under SYNTHETIC mean/covariance shifts, an energy-distance
permutation-test comparison on the same SYNTHETIC data, the Bonneel et al.
(2015) barycenter fixed point, and fail-closed edges. All SYNTHETIC:
correctness evidence only, never market evidence (AGENTS.md honesty
contract).
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from quant_fund.metrics.sliced_wasserstein import (
    deterministic_projections,
    max_sliced_wasserstein_distance,
    random_projections,
    sliced_wasserstein_barycenter,
    sliced_wasserstein_distance,
    sliced_wasserstein_test,
)
from quant_fund.metrics.wasserstein import (
    energy_distance_from_ot,
    wasserstein_1d,
    wasserstein_gaussian,
)

Array = NDArray[np.float64]

SEED = 20260929


def _rng() -> np.random.Generator:
    return np.random.default_rng(SEED)


def _clouds(n: int = 120, dim: int = 3, shift: float = 0.0) -> tuple[Array, Array]:
    rng = _rng()
    a = rng.normal(size=(n, dim))
    b = rng.normal(size=(n, dim))
    b[:, 0] += shift
    return a, b


def _energy_permutation_test(a: Array, b: Array, n_perm: int, seed: int) -> dict[str, float]:
    """Energy-distance permutation test on the pooled clouds (same calibration).

    Comparator for the SW test: the energy distance (Rizzo & Székely 2016;
    ``wasserstein.energy_distance_from_ot``) is also a sphere-average of a
    1-D discrepancy, so a head-to-head on identical SYNTHETIC data isolates
    the OT coupling as the only difference.
    """
    observed = energy_distance_from_ot(a, b)
    pooled = np.vstack((a, b))
    n_a = int(a.shape[0])
    n_tot = int(pooled.shape[0])
    rng = np.random.default_rng(seed)
    count_ge = 1
    for _ in range(n_perm):
        perm = rng.permutation(n_tot)
        stat = energy_distance_from_ot(pooled[perm[:n_a]], pooled[perm[n_a:]])
        if stat >= observed:
            count_ge += 1
    return {"statistic": float(observed), "pvalue": float(count_ge) / float(n_perm + 1)}


# ---------------------------------------------------------------- d=1 exactness


def test_sw_d1_equals_wasserstein_1d_exactly() -> None:
    rng = _rng()
    a = rng.normal(size=200)
    b = rng.normal(loc=0.5, scale=1.3, size=170)  # unequal sizes on purpose
    for p in (1.0, 2.0, 3.0):
        expected = wasserstein_1d(a, b, p=p)
        for mode in ("random", "deterministic"):
            for n_proj in (1, 7):
                got = sliced_wasserstein_distance(
                    a.reshape(-1, 1), b.reshape(-1, 1), p, n_proj, mode, SEED
                )
                assert got == pytest.approx(expected, abs=1e-12)


def test_msw_d1_equals_wasserstein_1d_exactly() -> None:
    rng = _rng()
    a = rng.normal(size=150).reshape(-1, 1)
    b = (rng.normal(size=150) + 2.0).reshape(-1, 1)
    for p in (1.0, 2.0):
        expected = wasserstein_1d(a[:, 0], b[:, 0], p=p)
        assert max_sliced_wasserstein_distance(a, b, p, 5, "random", SEED) == pytest.approx(
            expected, abs=1e-12
        )
        assert max_sliced_wasserstein_distance(a, b, p, 5, "deterministic", 0) == pytest.approx(
            expected, abs=1e-12
        )


# ---------------------------------------------------------------- zero self-distance


def test_self_distance_is_exactly_zero() -> None:
    rng = _rng()
    a = rng.normal(size=(80, 4))
    assert sliced_wasserstein_distance(a, a) == 0.0
    assert max_sliced_wasserstein_distance(a, a) == 0.0


def test_sw_positive_for_shifted_clouds() -> None:
    a, b = _clouds(shift=2.0)
    assert sliced_wasserstein_distance(a, b, seed=SEED) > 0.0


# ---------------------------------------------------------------- known bounds


def test_sw_le_w2_gaussian_closed_form() -> None:
    # SW_p <= W_p with constant 1 (1-Lipschitz projections); anchor W2 with
    # the Gaussian closed form (Gelbrich 1990) on a mean shift.
    dim = 3
    rng = _rng()
    a = rng.normal(size=(4000, dim))
    b = rng.normal(size=(4000, dim)) + np.array([2.0, -1.0, 0.5])
    sw2 = sliced_wasserstein_distance(a, b, 2.0, 256, "random", SEED)
    w2_sq = wasserstein_gaussian(
        np.zeros(dim), np.eye(dim), np.array([2.0, -1.0, 0.5]), np.eye(dim)
    )
    assert sw2 <= np.sqrt(w2_sq) + 0.05  # population W2 plus MC slack


def test_w2_equals_sqrt_d_times_sw_for_isotropic_scale_shift() -> None:
    # Sharpness of the reverse bound: for N(0, I_d) vs N(0, s^2 I_d) every
    # 1-D marginal is N(0,1) vs N(0,s^2), so SW_2 = s - 1 direction-free and
    # W_2^2 = d (s-1)^2 — i.e. W_2 = sqrt(d) * SW_2 exactly (Nadjahi et al.
    # 2021: the sqrt(d) constant is attained).
    dim = 4
    s = 1.5
    rng = _rng()
    a = rng.normal(size=(6000, dim))
    b = s * rng.normal(size=(6000, dim))
    sw2 = sliced_wasserstein_distance(a, b, 2.0, 128, "random", SEED)
    w2_sq = wasserstein_gaussian(np.zeros(dim), np.eye(dim), np.zeros(dim), (s * s) * np.eye(dim))
    assert sw2 == pytest.approx(s - 1.0, abs=0.03)
    assert sw2 * sw2 * float(dim) == pytest.approx(w2_sq, abs=0.15)
    assert sw2 <= np.sqrt(w2_sq) + 1e-9  # SW_2 <= W_2 for d >= 1


def test_sw_le_msw_on_common_directions() -> None:
    a, b = _clouds(n=200, dim=4, shift=0.5)
    sw = sliced_wasserstein_distance(a, b, 2.0, 64, "random", SEED)
    msw = max_sliced_wasserstein_distance(a, b, 2.0, 64, "random", SEED)
    assert sw <= msw + 1e-12
    # MSW also stays under the population W2 of the mean shift.
    w2_sq = wasserstein_gaussian(np.zeros(4), np.eye(4), np.array([0.5, 0, 0, 0]), np.eye(4))
    assert msw <= np.sqrt(w2_sq) + 0.15


def test_sw_symmetric() -> None:
    a, b = _clouds(shift=1.0)
    assert sliced_wasserstein_distance(a, b, seed=SEED) == pytest.approx(
        sliced_wasserstein_distance(b, a, seed=SEED), abs=1e-12
    )


def test_sw_triangle_inequality_deterministic() -> None:
    # With a fixed common direction set SW_p is a genuine metric (Minkowski
    # over the per-direction W_p), so the triangle inequality holds exactly
    # up to floating point.
    rng = _rng()
    a = rng.normal(size=(100, 2))
    b = rng.normal(size=(100, 2)) + 1.0
    c = rng.normal(size=(100, 2)) * 1.5
    d_ab = sliced_wasserstein_distance(a, b, p=2.0, n_projections=32, projection="deterministic")
    d_bc = sliced_wasserstein_distance(b, c, p=2.0, n_projections=32, projection="deterministic")
    d_ac = sliced_wasserstein_distance(a, c, p=2.0, n_projections=32, projection="deterministic")
    assert d_ac <= d_ab + d_bc + 1e-12


# ---------------------------------------------------------------- determinism


def test_determinism_pinned_same_seed_and_projection_mode() -> None:
    a, b = _clouds(shift=0.75)
    d1 = sliced_wasserstein_distance(a, b, 2.0, 48, "random", SEED)
    d2 = sliced_wasserstein_distance(a, b, 2.0, 48, "random", SEED)
    assert d1 == d2  # bit-identical, not merely close
    assert sliced_wasserstein_distance(a, b, 2.0, 48, "random", SEED + 1) != d1
    # Deterministic mode ignores the seed entirely.
    e1 = sliced_wasserstein_distance(a, b, 2.0, 48, "deterministic", 0)
    e2 = sliced_wasserstein_distance(a, b, 2.0, 48, "deterministic", 123)
    assert e1 == e2
    t1 = sliced_wasserstein_test(a, b, "sw", 2.0, 16, "random", SEED, n_perm=9)
    t2 = sliced_wasserstein_test(a, b, "sw", 2.0, 16, "random", SEED, n_perm=9)
    assert t1 == t2


def test_projection_directions_are_unit_and_reproducible() -> None:
    th = random_projections(5, 16, SEED)
    assert th.shape == (16, 5)
    assert np.allclose(np.linalg.norm(th, axis=1), 1.0, atol=1e-12)
    assert np.array_equal(th, random_projections(5, 16, SEED))
    th2 = deterministic_projections(2, 8)
    assert np.allclose(np.linalg.norm(th2, axis=1), 1.0, atol=1e-12)
    assert np.array_equal(th2, deterministic_projections(2, 8))
    th3 = deterministic_projections(4, 6)
    assert th3.shape == (6, 4)
    assert np.allclose(np.linalg.norm(th3, axis=1), 1.0, atol=1e-12)
    assert np.array_equal(th3, deterministic_projections(4, 6))


# ---------------------------------------------------------------- test statistic


def test_test_statistic_matches_public_distance() -> None:
    # Pins the batched quantile-grid replica in the permutation loop to the
    # wasserstein_1d-based public distance: same seed/directions -> same stat.
    a, b = _clouds(n=150, dim=3, shift=0.5)
    res = sliced_wasserstein_test(a, b, "sw", 2.0, 32, "random", SEED, n_perm=1)
    dist = sliced_wasserstein_distance(a, b, 2.0, 32, "random", SEED)
    assert res["statistic"] == pytest.approx(dist, abs=1e-12)
    res_m = sliced_wasserstein_test(a, b, "msw", 2.0, 32, "random", SEED, n_perm=1)
    dist_m = max_sliced_wasserstein_distance(a, b, 2.0, 32, "random", SEED)
    assert res_m["statistic"] == pytest.approx(dist_m, abs=1e-12)


def test_identical_clouds_give_pvalue_one() -> None:
    rng = _rng()
    a = rng.normal(size=(100, 2))
    res = sliced_wasserstein_test(a, a, "sw", 2.0, 16, "random", SEED, n_perm=49)
    assert res["statistic"] == 0.0
    assert res["pvalue"] == 1.0  # every permutation stat >= 0 = observed


def test_size_under_null_is_near_alpha() -> None:
    # SYNTHETIC null: A, B iid from the same N(0, I_3). A calibrated
    # permutation p-value rejects at rate <= alpha; across 20 seeded trials
    # at alpha = 0.05 the expected rejection count is 1.0, so >= 5 would be
    # a ~0.3% binomial tail event and indicates size distortion.
    rejects = 0
    trials = 20
    for trial in range(trials):
        rng = np.random.default_rng(SEED + trial)
        a = rng.normal(size=(50, 3))
        b = rng.normal(size=(50, 3))
        res = sliced_wasserstein_test(a, b, "sw", 2.0, 24, "random", SEED + 1000 + trial, n_perm=99)
        if res["pvalue"] <= 0.05:
            rejects += 1
    assert rejects <= 4


def test_power_under_mean_shift() -> None:
    # SYNTHETIC mean shift of 1 sigma along one coordinate in d=4.
    a, b = _clouds(n=150, dim=4, shift=1.0)
    res = sliced_wasserstein_test(a, b, "sw", 2.0, 64, "random", SEED, n_perm=299)
    assert res["statistic"] > 0.0
    assert res["pvalue"] <= 0.01
    res_m = sliced_wasserstein_test(a, b, "msw", 2.0, 64, "random", SEED, n_perm=299)
    assert res_m["pvalue"] <= 0.01


def test_power_under_covariance_shift() -> None:
    # SYNTHETIC covariance shift: same mean, diag variances (2.0, 0.6, 1.0).
    rng = _rng()
    a = rng.normal(size=(200, 3))
    b = rng.normal(size=(200, 3)) * np.array([np.sqrt(2.0), np.sqrt(0.6), 1.0])
    res = sliced_wasserstein_test(a, b, "sw", 2.0, 64, "random", SEED, n_perm=299)
    assert res["pvalue"] <= 0.01


def test_pvalue_never_zero_and_bounded() -> None:
    # Phipson & Smyth (2010): pvalue in [1/(n_perm+1), 1] by construction.
    a, b = _clouds(n=60, dim=2, shift=3.0)  # extreme shift: all perms smaller
    res = sliced_wasserstein_test(a, b, "sw", 2.0, 16, "random", SEED, n_perm=99)
    assert res["pvalue"] >= 1.0 / 100.0
    assert res["pvalue"] <= 1.0
    assert res["pvalue"] == pytest.approx(0.01)  # only the observed split counts


# ---------------------------------------------------------------- SW vs energy bridge


def test_sw_vs_energy_size_under_null() -> None:
    # Same SYNTHETIC null pair, same calibration: both tests must not reject
    # at 5% (seeds pinned, so this is a deterministic check).
    rng = _rng()
    a = rng.normal(size=(120, 3))
    b = rng.normal(size=(120, 3))
    res_sw = sliced_wasserstein_test(a, b, "sw", 2.0, 48, "random", SEED, n_perm=199)
    res_en = _energy_permutation_test(a, b, n_perm=199, seed=SEED)
    assert res_sw["pvalue"] > 0.05
    assert res_en["pvalue"] > 0.05


def test_sw_and_energy_both_detect_mean_shift() -> None:
    # SYNTHETIC mean shift: both statistics are consistent (Ramdas et al.
    # 2017; Rizzo & Székely 2016), so both must reject on a strong shift.
    a, b = _clouds(n=150, dim=4, shift=1.5)
    res_sw = sliced_wasserstein_test(a, b, "sw", 2.0, 64, "random", SEED, n_perm=199)
    res_en = _energy_permutation_test(a, b, n_perm=199, seed=SEED)
    assert res_sw["pvalue"] <= 0.05
    assert res_en["pvalue"] <= 0.05


def test_sw_and_energy_both_detect_covariance_shift() -> None:
    # SYNTHETIC covariance shift: SW sees it through the 1-D marginals of
    # (almost) every direction; energy sees it through the pairwise-distance
    # distribution. Both must reject.
    rng = _rng()
    a = rng.normal(size=(200, 3))
    b = rng.normal(size=(200, 3)) * np.array([1.8, 0.7, 1.0])
    res_sw = sliced_wasserstein_test(a, b, "sw", 2.0, 64, "random", SEED, n_perm=199)
    res_en = _energy_permutation_test(a, b, n_perm=199, seed=SEED)
    assert res_sw["pvalue"] <= 0.05
    assert res_en["pvalue"] <= 0.05


# ---------------------------------------------------------------- barycenter


def test_barycenter_of_identical_inputs_returns_that_input() -> None:
    rng = _rng()
    x = rng.normal(size=(40, 2))
    for mode in ("random", "deterministic"):
        res = sliced_wasserstein_barycenter(
            [x, x.copy(), x.copy()], n_projections=16, projection=mode, seed=SEED
        )
        assert res.points == pytest.approx(x, abs=1e-12)
        assert res.converged
        assert res.n_iter == 1
        assert res.final_delta <= 1e-12


def test_barycenter_single_cloud_is_that_cloud() -> None:
    rng = _rng()
    x = rng.normal(size=(30, 3))
    res = sliced_wasserstein_barycenter([x], n_projections=8, seed=SEED)
    assert res.points == pytest.approx(x, abs=1e-12)
    assert res.converged


def test_barycenter_of_two_point_masses_is_exact_midpoint() -> None:
    n = 25
    x1 = np.zeros((n, 2))
    x2 = np.full((n, 2), 4.0)
    res = sliced_wasserstein_barycenter(
        [x1, x2], n_projections=16, projection="deterministic", seed=0
    )
    assert res.points == pytest.approx(2.0, abs=1e-12)
    assert res.converged
    # Weighted version moves the barycenter to the weighted midpoint.
    res_w = sliced_wasserstein_barycenter(
        [x1, x2], weights=np.array([0.25, 0.75]), n_projections=16, projection="deterministic"
    )
    assert res_w.points == pytest.approx(3.0, abs=1e-12)


def test_barycenter_preserves_weighted_input_mean_exactly() -> None:
    # Rank matching only permutes points within each cloud, so the barycenter
    # mean is the weighted mean of the input means at every sweep — an exact
    # invariant of the Bonneel et al. (2015) fixed point, whether or not the
    # iteration reaches its tol (see the limit-cycle note below).
    rng = _rng()
    clouds = [rng.normal(loc=k, scale=1.0 + 0.3 * k, size=(60, 3)) for k in range(3)]
    w = np.array([0.2, 0.3, 0.5])
    res = sliced_wasserstein_barycenter(clouds, weights=w, n_projections=32, seed=SEED, max_iter=50)
    expected_mean = sum(w[k] * clouds[k].mean(axis=0) for k in range(3))
    assert res.points.mean(axis=0) == pytest.approx(expected_mean, abs=1e-10)


def test_barycenter_d1_converges_exactly_to_quantile_average() -> None:
    # In d=1 every direction is +/- the identity, the per-direction rank
    # constraints are consistent, and the SW barycenter is exactly the
    # weighted quantile average — the fixed point must converge to tol in a
    # couple of sweeps and return that average.
    rng = _rng()
    a = rng.normal(size=(80, 1))
    b = rng.normal(loc=1.0, size=(80, 1))
    for mode in ("deterministic", "random"):
        res = sliced_wasserstein_barycenter(
            [a, b], n_projections=8, projection=mode, seed=SEED, max_iter=50
        )
        assert res.converged
        assert res.n_iter <= 2
        expected = 0.5 * (np.sort(a[:, 0]) + np.sort(b[:, 0]))
        assert np.sort(res.points[:, 0]) == pytest.approx(expected, abs=1e-12)


def test_barycenter_of_symmetric_shifted_gaussians_centers_at_zero() -> None:
    # SYNTHETIC: N(+mu, I) and N(-mu, I), equal weights, dim=3. The
    # equal-weight barycenter must sit between the parents (mean ~ 0, exact
    # invariant) and stay roughly equidistant (in SW) from both. With a
    # finite random direction set in dim >= 2 the rank constraints are
    # mutually inconsistent, so the sweep-level displacement stays at
    # limit-cycle level and the convergence check must honestly report
    # converged=False rather than fake a fixed point.
    dim = 3
    mu = np.array([1.5, -1.0, 0.5])
    rng = _rng()
    x1 = rng.normal(size=(150, dim)) + mu
    x2 = rng.normal(size=(150, dim)) - mu
    res = sliced_wasserstein_barycenter([x1, x2], n_projections=64, seed=SEED, max_iter=200)
    assert not res.converged  # honest limit-cycle reporting (tol=1e-8 unreachable)
    assert res.final_delta > 1e-8
    # Exact invariant: barycenter mean = weighted mean of the input means,
    # which here is the (small, sampling-noise) midpoint of +mu/-mu draws.
    mid = 0.5 * (x1.mean(axis=0) + x2.mean(axis=0))
    assert res.points.mean(axis=0) == pytest.approx(mid, abs=1e-10)
    assert np.max(np.abs(mid)) < 0.25  # sits between the parents, not at either
    d1 = sliced_wasserstein_distance(res.points, x1, 2.0, 128, "random", SEED)
    d2 = sliced_wasserstein_distance(res.points, x2, 2.0, 128, "random", SEED)
    assert d1 == pytest.approx(d2, rel=0.25)


def test_barycenter_determinism_pinned() -> None:
    rng = _rng()
    clouds = [rng.normal(size=(50, 2)), rng.normal(size=(50, 2)) + 1.0]
    r1 = sliced_wasserstein_barycenter(clouds, n_projections=24, seed=SEED)
    r2 = sliced_wasserstein_barycenter(clouds, n_projections=24, seed=SEED)
    assert np.array_equal(r1.points, r2.points)
    assert r1.n_iter == r2.n_iter
    assert r1.final_delta == r2.final_delta


# ---------------------------------------------------------------- fail-closed edges


def test_distance_fail_closed() -> None:
    a, b = _clouds(n=20, dim=2)
    c = _rng().normal(size=(20, 3))
    with pytest.raises(ValueError):
        sliced_wasserstein_distance(a, c)  # dimension mismatch
    with pytest.raises(ValueError):
        sliced_wasserstein_distance(a[:0], a)  # empty
    with pytest.raises(ValueError):
        sliced_wasserstein_distance(np.array([[np.nan, 0.0]]), a)  # non-finite
    with pytest.raises(ValueError):
        sliced_wasserstein_distance(a, b, p=0.5)  # p < 1
    with pytest.raises(ValueError):
        sliced_wasserstein_distance(a, b, p=np.nan)
    with pytest.raises(ValueError):
        sliced_wasserstein_distance(a, b, n_projections=0)  # L = 0
    with pytest.raises(ValueError):
        sliced_wasserstein_distance(a, b, projection="bogus")
    with pytest.raises(ValueError):
        sliced_wasserstein_distance(a, b, seed=-1)
    with pytest.raises(ValueError):
        max_sliced_wasserstein_distance(a, c)
    with pytest.raises(ValueError):
        max_sliced_wasserstein_distance(a, b, n_projections=-3)
    with pytest.raises(ValueError):
        random_projections(2, 0, SEED)
    with pytest.raises(ValueError):
        deterministic_projections(0, 4)


def test_test_fail_closed() -> None:
    a, b = _clouds(n=20, dim=2)
    c = _rng().normal(size=(20, 3))
    with pytest.raises(ValueError):
        sliced_wasserstein_test(a, c)  # dimension mismatch
    with pytest.raises(ValueError):
        sliced_wasserstein_test(a, b, statistic="bogus")
    with pytest.raises(ValueError):
        sliced_wasserstein_test(a, b, n_perm=0)
    with pytest.raises(ValueError):
        sliced_wasserstein_test(a, b, n_projections=0)
    with pytest.raises(ValueError):
        sliced_wasserstein_test(a, b, p=0.5)
    with pytest.raises(ValueError):
        sliced_wasserstein_test(a[:0], b)


def test_barycenter_fail_closed() -> None:
    rng = _rng()
    x = rng.normal(size=(10, 2))
    y = rng.normal(size=(10, 3))
    z9 = rng.normal(size=(9, 2))
    with pytest.raises(ValueError):
        sliced_wasserstein_barycenter([])  # empty list
    with pytest.raises(ValueError):
        sliced_wasserstein_barycenter([x, y])  # dimension mismatch
    with pytest.raises(ValueError):
        sliced_wasserstein_barycenter([x, z9])  # cardinality mismatch
    with pytest.raises(ValueError):
        sliced_wasserstein_barycenter([x, x], weights=np.array([1.0]))  # wrong length
    with pytest.raises(ValueError):
        sliced_wasserstein_barycenter([x, x], weights=np.array([1.0, -1.0]))  # negative
    with pytest.raises(ValueError):
        sliced_wasserstein_barycenter([x, x], weights=np.zeros(2))  # zero total
    with pytest.raises(ValueError):
        sliced_wasserstein_barycenter([x, x], weights=np.array([np.nan, 1.0]))
    with pytest.raises(ValueError):
        sliced_wasserstein_barycenter([x], n_projections=0)
    with pytest.raises(ValueError):
        sliced_wasserstein_barycenter([x], projection="bogus")
    with pytest.raises(ValueError):
        sliced_wasserstein_barycenter([x], max_iter=0)
    with pytest.raises(ValueError):
        sliced_wasserstein_barycenter([x], tol=0.0)
    with pytest.raises(ValueError):
        sliced_wasserstein_barycenter([np.full((5, 2), np.inf)])  # non-finite
