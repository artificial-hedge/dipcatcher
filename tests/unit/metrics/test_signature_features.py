"""Signature-feature metrics — SYNTHETIC correctness tests.

Validates the wave-21 layer on top of ``models.path_signatures``:
dyadic-refined Goursat-PDE signature kernel (Salvi, Cass, Foster, Lyons &
Yang 2021, arXiv:2006.14794), its convergence to the truncated signature
inner product and the straight-line closed form
``I_0(2·sqrt(⟨V_a, V_b⟩))``, channel augmentations, batched feature
matrices, Gram PSD sanity, and the signature-kernel MMD two-sample test
(Chevyrev & Oberhauser 2018, arXiv:1810.10971; Gretton et al. 2008/2012,
arXiv:0805.2368). All ensembles are seeded SYNTHETIC paths — correctness
evidence only, never market evidence (AGENTS.md honesty contract #2).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.special import i0

from quant_fund.metrics.signature_features import (
    augment_path,
    bench_signature_features,
    dyadic_refine,
    signature_feature_matrix,
    signature_feature_vector,
    signature_gram,
    signature_kernel_pde,
    signature_kernel_pde_diagnostics,
    signature_mmd,
    signature_mmd_test,
    time_augmentation,
)
from quant_fund.models.path_signatures import (
    lead_lag_transform,
    logsignature,
    signature,
    signature_kernel,
)

Array = np.ndarray

BENCH_SEED = 3  # kept outside the tuning seeds (0, 1, 2, 7) used while calibrating


def _line(v: float, dims_pos: int = 0, dims: int = 1) -> Array:
    """One-segment straight path with total displacement v in channel dims_pos."""
    p = np.zeros((2, dims))
    p[1, dims_pos] = v
    return p


def _random_walk(rng: np.random.Generator, n_inc: int, dims: int, sd: float = 0.25) -> Array:
    increments = rng.normal(0.0, sd, (n_inc, dims))
    return np.concatenate([np.zeros((1, dims)), np.cumsum(increments, axis=0)])


def _gbm_batch(
    rng: np.random.Generator, n: int, steps: int, sigma: float, mu: float = 0.0
) -> list[Array]:
    """n time-augmented drifted-BM paths (SYNTHETIC ensemble helper)."""
    dt = 1.0 / steps
    inc = rng.normal(mu * dt, sigma * math.sqrt(dt), (n, steps))
    paths = np.concatenate([np.zeros((n, 1, 1)), np.cumsum(inc, axis=1)[:, :, None]], axis=1)
    return [time_augmentation(paths[i]) for i in range(n)]


def _ou_batch(
    rng: np.random.Generator, n: int, steps: int, theta: float, sigma: float
) -> list[Array]:
    dt = 1.0 / steps
    shocks = rng.normal(0.0, sigma * math.sqrt(dt), (n, steps))
    out = np.zeros((n, steps + 1))
    for t in range(steps):
        out[:, t + 1] = out[:, t] - theta * out[:, t] * dt + shocks[:, t]
    return [time_augmentation(out[i, :, None]) for i in range(n)]


# ---------------------------------------------------------------------------
# time_augmentation / augment_path
# ---------------------------------------------------------------------------


def test_time_augmentation_appends_normalized_clock() -> None:
    path = np.array([[0.0, 1.0], [0.5, 1.5], [1.0, 0.5]])
    aug = time_augmentation(path)
    assert aug.shape == (3, 3)
    np.testing.assert_allclose(aug[:, :2], path, rtol=0, atol=0)
    np.testing.assert_allclose(aug[:, 2], [0.0, 0.5, 1.0], rtol=0, atol=0)


def test_augment_path_leadlag_doubles_channels_and_matches_upstream() -> None:
    path = np.array([[0.0], [0.3], [0.1], [0.7]])
    out = augment_path(path, augmentations=("leadlag",))
    np.testing.assert_allclose(out, lead_lag_transform(path), rtol=0, atol=0)
    assert out.shape[1] == 2 * path.shape[1]


def test_augment_path_order_applies_left_to_right() -> None:
    path = np.array([[0.0], [0.3], [0.1]])
    both = augment_path(path, augmentations=("time", "leadlag"))
    expected = lead_lag_transform(time_augmentation(path))
    np.testing.assert_allclose(both, expected, rtol=0, atol=0)
    # clock first, then lead-lag: (T+1, d+1) -> (2T+1, 2(d+1))
    assert both.shape == (2 * path.shape[0] - 1, 2 * (path.shape[1] + 1))


def test_augment_path_empty_tuple_and_bad_name() -> None:
    path = np.array([[0.0], [0.3], [0.1]])
    np.testing.assert_allclose(augment_path(path, augmentations=()), path, rtol=0, atol=0)
    with pytest.raises(ValueError, match="augmentation"):
        augment_path(path, augmentations=("warp",))


# ---------------------------------------------------------------------------
# feature vectors / matrices
# ---------------------------------------------------------------------------


def test_feature_vector_matches_upstream_signature() -> None:
    rng = np.random.default_rng(5)
    path = _random_walk(rng, 20, 2)
    vec = signature_feature_vector(path, order=3, augmentations=())
    np.testing.assert_allclose(vec, signature(path, 3), rtol=0, atol=0)
    # default adds the time channel first
    vec_t = signature_feature_vector(path, order=3)
    np.testing.assert_allclose(vec_t, signature(time_augmentation(path), 3), rtol=0, atol=0)


def test_feature_vector_level1_is_net_displacement() -> None:
    path = np.array([[1.0, 2.0], [1.5, 2.2], [-0.5, 3.0]])
    vec = signature_feature_vector(path, order=2, augmentations=())
    np.testing.assert_allclose(vec[:2], [-1.5, 1.0], rtol=1e-12)


def test_logsignature_basis_dims_and_order_bound() -> None:
    rng = np.random.default_rng(9)
    path = _random_walk(rng, 15, 1)
    # 1-D path + time channel -> d = 2; Witt dims l1=2, l2=1, l3=2 -> 5
    vec = signature_feature_vector(path, order=3, basis="logsignature")
    assert vec.shape == (5,)
    np.testing.assert_allclose(vec, logsignature(time_augmentation(path), 3), rtol=0, atol=0)
    with pytest.raises(ValueError, match="order"):
        signature_feature_vector(path, order=5, basis="logsignature")


def test_feature_vector_rejects_bad_basis_and_order() -> None:
    path = np.array([[0.0], [1.0]])
    with pytest.raises(ValueError, match="basis"):
        signature_feature_vector(path, basis="fancy")
    with pytest.raises(ValueError, match="order"):
        signature_feature_vector(path, order=0)
    with pytest.raises(ValueError, match="order"):
        signature_feature_vector(path, order=7)


def test_feature_matrix_stacks_and_checks_ensemble() -> None:
    rng = np.random.default_rng(11)
    paths = [_random_walk(rng, 12, 1) for _ in range(4)]
    # default time augmentation lifts d=1 -> d=2; signature order 3 has
    # 2 + 4 + 8 = 14 word-ordered columns
    mat = signature_feature_matrix(paths, order=3, basis="signature")
    assert mat.shape == (4, 14)
    np.testing.assert_allclose(mat[0], signature(time_augmentation(paths[0]), 3), rtol=0)


def test_feature_matrix_channel_mismatch_and_empty_fail() -> None:
    a = np.array([[0.0, 0.0], [1.0, 1.0]])
    b = np.array([[0.0], [1.0]])
    with pytest.raises(ValueError, match="channel"):
        signature_feature_matrix([a, b])
    with pytest.raises(ValueError, match="at least 1"):
        signature_feature_matrix([])


def test_path_validation_rejects_rank_and_nan() -> None:
    with pytest.raises(ValueError, match="2-D"):
        signature_feature_vector(np.array([0.0, 1.0, 2.0]))
    with pytest.raises(ValueError, match="at least 2 points"):
        signature_feature_vector(np.array([[0.0, 0.0]]))
    with pytest.raises(ValueError, match="finite"):
        signature_feature_vector(np.array([[0.0], [np.nan]]))
    with pytest.raises(ValueError, match="finite"):
        signature_kernel_pde(np.array([[0.0], [1.0]]), np.array([[0.0], [np.inf]]))


# ---------------------------------------------------------------------------
# dyadic_refine
# ---------------------------------------------------------------------------


def test_dyadic_refine_level0_is_copy() -> None:
    path = np.array([[0.0], [1.0], [0.5]])
    out = dyadic_refine(path, 0)
    np.testing.assert_allclose(out, path, rtol=0, atol=0)
    assert out is not path


def test_dyadic_refine_inserts_collinear_midpoints() -> None:
    path = np.array([[0.0, 0.0], [2.0, 4.0], [4.0, 4.0]])
    r1 = dyadic_refine(path, 1)
    assert r1.shape == (2 * (path.shape[0] - 1) + 1, 2)
    np.testing.assert_allclose(r1[1], [1.0, 2.0], rtol=0, atol=0)  # midpoint seg 1
    np.testing.assert_allclose(r1[3], [3.0, 4.0], rtol=0, atol=0)  # midpoint seg 2
    np.testing.assert_allclose(r1[[0, 2, 4]], path, rtol=0, atol=0)


def test_dyadic_refine_preserves_signature_exactly() -> None:
    rng = np.random.default_rng(13)
    path = _random_walk(rng, 9, 2)
    for level in (1, 2, 3):
        np.testing.assert_allclose(
            signature(dyadic_refine(path, level), 4),
            signature(path, 4),
            rtol=1e-12,
            atol=1e-12,
        )


def test_dyadic_refine_bad_levels_fail_closed() -> None:
    path = np.array([[0.0], [1.0]])
    for bad in (-1, 9, 1.5, "x"):
        with pytest.raises(ValueError):
            dyadic_refine(path, bad)  # type: ignore[arg-type]
    long_path = np.linspace(0, 1, 5000)[:, None]
    with pytest.raises(ValueError, match="segments"):
        dyadic_refine(long_path, 3)


# ---------------------------------------------------------------------------
# signature_kernel_pde
# ---------------------------------------------------------------------------


def test_pde_kernel_straight_line_closed_form() -> None:
    # linear paths: k = sum_k a^k/(k!)^2 = I_0(2 sqrt(a))
    a_dot = 0.6 * 1.0
    exact = float(i0(2.0 * math.sqrt(a_dot)))
    est = signature_kernel_pde(_line(1.0), _line(0.6), dyadic_level=2)
    assert est == pytest.approx(exact, abs=2e-3)
    # negative inner product (opposite directions): the level sum
    # sum_k a^k/(k!)^2 still converges (it is J_0(2 sqrt(|a|)) in Bessel
    # terms) — evaluate the series directly as the closed-form reference.
    est_neg = signature_kernel_pde(_line(1.0), _line(-0.5), dyadic_level=2)
    series = sum((-0.5) ** k / math.factorial(k) ** 2 for k in range(12))
    assert est_neg == pytest.approx(series, abs=2e-3)


def test_pde_kernel_constant_path_is_one() -> None:
    flat = np.zeros((7, 2))
    assert signature_kernel_pde(flat, flat, dyadic_level=0) == pytest.approx(1.0)
    other = np.array([[0.0, 0.0], [0.5, -0.2]])
    assert signature_kernel_pde(flat, other, dyadic_level=1) == pytest.approx(1.0)


def test_pde_kernel_self_is_norm_squared() -> None:
    rng = np.random.default_rng(17)
    path = _random_walk(rng, 16, 2)
    kxx = signature_kernel_pde(path, path, dyadic_level=0)
    assert kxx >= 1.0  # level-0 term alone contributes 1
    # diagonal of a Gram equals self kernel
    gram = signature_gram([path, _random_walk(rng, 12, 2)], dyadic_level=0)
    assert gram[0, 0] == pytest.approx(kxx)


def test_pde_kernel_symmetric_and_sigma_scaling() -> None:
    rng = np.random.default_rng(19)
    a = _random_walk(rng, 18, 2)
    b = _random_walk(rng, 14, 2)
    k_ab = signature_kernel_pde(a, b, dyadic_level=1)
    k_ba = signature_kernel_pde(b, a, dyadic_level=1)
    assert k_ab == pytest.approx(k_ba, abs=1e-12)
    # K_sigma(a, b) = K_1(sigma a, sigma b): increments scale the cell terms
    k_scaled = signature_kernel_pde(a, b, dyadic_level=1, sigma=0.5)
    k_scaled_ref = signature_kernel_pde(0.5 * a, 0.5 * b, dyadic_level=1, sigma=1.0)
    assert k_scaled == pytest.approx(k_scaled_ref, rel=1e-12)


def test_pde_kernel_converges_to_truncated_inner_product() -> None:
    rng = np.random.default_rng(23)
    a = _random_walk(rng, 20, 2, sd=0.15)
    b = _random_walk(rng, 24, 2, sd=0.15)
    pde = signature_kernel_pde(a, b, dyadic_level=2)
    ref = signature_kernel(a, b, order=6)
    assert pde == pytest.approx(ref, abs=2e-2)


def test_pde_diagnostics_shrinking_gaps_and_reference() -> None:
    rng = np.random.default_rng(29)
    a = _random_walk(rng, 14, 2, sd=0.2)
    b = _random_walk(rng, 14, 2, sd=0.2)
    diag = signature_kernel_pde_diagnostics(a, b, max_dyadic_level=3)
    assert len(diag["estimates"]) == 4
    assert len(diag["gaps"]) == 3
    assert isinstance(diag["monotone_shrinking_gaps"], bool)
    # the refined estimate sits closer to the order-6 truncated reference
    # than the coarsest grid does
    coarse_gap = abs(diag["estimates"][0] - diag["truncated_reference_order6"])
    assert diag["abs_gap_to_reference"] <= coarse_gap + 1e-9
    assert diag["abs_gap_to_reference"] < 0.5
    assert diag["final"] == diag["estimates"][-1]


def test_pde_kernel_detects_path_ordering() -> None:
    # time-shuffled increments give a different kernel value
    rng = np.random.default_rng(31)
    base = _random_walk(rng, 24, 2)
    perm = rng.permutation(base.shape[0])
    shuffled = np.concatenate([base[:1], base[perm[1:]]])
    k_same = signature_kernel_pde(base, base, dyadic_level=0)
    k_shuf = signature_kernel_pde(base, shuffled, dyadic_level=0)
    assert k_same != pytest.approx(k_shuf, rel=1e-9)


def test_pde_kernel_fail_closed_edges() -> None:
    a2 = np.array([[0.0, 0.0], [1.0, 1.0]])
    a1 = np.array([[0.0], [1.0]])
    with pytest.raises(ValueError, match="channel"):
        signature_kernel_pde(a2, a1)
    with pytest.raises(ValueError, match="scheme"):
        signature_kernel_pde(a2, a2, scheme="pde3")
    with pytest.raises(ValueError, match="sigma"):
        signature_kernel_pde(a2, a2, sigma=0.0)
    with pytest.raises(ValueError):
        signature_kernel_pde(a2, a2, dyadic_level=-1)
    # pde1 denominator blows up on a cell with <dx, dy> = 2
    big_a = np.array([[0.0], [math.sqrt(2.0)]])
    with pytest.raises(ArithmeticError):
        signature_kernel_pde(big_a, big_a, dyadic_level=0, scheme="pde1")


# ---------------------------------------------------------------------------
# signature_gram / mmd
# ---------------------------------------------------------------------------


def test_gram_symmetric_and_matches_pairwise() -> None:
    rng = np.random.default_rng(37)
    paths = [_random_walk(rng, 12, 2, sd=0.2) for _ in range(4)]
    gram = signature_gram(paths, dyadic_level=0)
    np.testing.assert_allclose(gram, gram.T, rtol=0, atol=0)
    for i in range(4):
        for j in range(4):
            assert gram[i, j] == pytest.approx(
                signature_kernel_pde(paths[i], paths[j], dyadic_level=0), rel=1e-12
            )


def test_gram_cross_shape_and_channel_guard() -> None:
    rng = np.random.default_rng(41)
    a = [_random_walk(rng, 10, 2, sd=0.2) for _ in range(3)]
    b = [_random_walk(rng, 8, 2, sd=0.2) for _ in range(5)]
    gram = signature_gram(a, b, dyadic_level=0)
    assert gram.shape == (3, 5)
    b_bad = [_random_walk(rng, 8, 1, sd=0.2) for _ in range(2)]
    with pytest.raises(ValueError, match="channel"):
        signature_gram(a, b_bad)


def test_gram_truncated_matches_upstream_kernel() -> None:
    rng = np.random.default_rng(43)
    paths = [_random_walk(rng, 11, 2, sd=0.3) for _ in range(3)]
    gram = signature_gram(paths, kernel="truncated", order=3, sigma=0.7)
    for i in range(3):
        for j in range(3):
            assert gram[i, j] == pytest.approx(
                signature_kernel(paths[i], paths[j], 3, 0.7), abs=1e-10
            )
    # PSD: truncated kernel is a plain feature-space inner product
    assert float(np.linalg.eigvalsh(gram).min()) >= -1e-9


def test_gram_rejects_unknown_kernel_and_bad_order() -> None:
    paths = [np.array([[0.0], [1.0]]), np.array([[0.0], [0.5]])]
    with pytest.raises(ValueError, match="kernel"):
        signature_gram(paths, kernel="gibbs")
    with pytest.raises(ValueError, match="order"):
        signature_gram(paths, kernel="truncated", order=9)


def test_mmd_identical_ensembles_near_zero() -> None:
    rng = np.random.default_rng(47)
    a = _gbm_batch(rng, 8, 24, 0.8)
    res = signature_mmd(a, [p.copy() for p in a], kernel="truncated", order=3)
    assert res["mmd2_biased"] == pytest.approx(0.0, abs=1e-12)
    # the unbiased U-statistic drops self-kernel diagonals — for the
    # signature kernel K(x,x) = ||Sig(x)||^2 dominates the Gram, so on
    # identical ensembles it legitimately goes negative (never clipped).
    assert res["mmd2_unbiased"] <= 1e-9
    assert math.isfinite(res["mmd2_unbiased"])
    assert res["n_a"] == 8.0 and res["n_b"] == 8.0


def test_mmd_biased_statistic_nonnegative() -> None:
    rng = np.random.default_rng(53)
    gbm = _gbm_batch(rng, 10, 32, 0.9)
    ou = _ou_batch(rng, 10, 32, 8.0, 0.5)
    same = signature_mmd(gbm, _gbm_batch(rng, 10, 32, 0.9), kernel="truncated", order=3, sigma=2.0)
    diff = signature_mmd(gbm, ou, kernel="truncated", order=3, sigma=2.0)
    # at small n the biased V-statistic is noisy and does not itself
    # discriminate ensembles — that claim lives in the permutation test
    # (signature_mmd_test) and the bench power metric, not the raw score.
    assert same["mmd2_biased"] >= -1e-12
    assert diff["mmd2_biased"] >= -1e-12


def test_mmd_fail_closed_small_ensembles_and_mismatch() -> None:
    rng = np.random.default_rng(59)
    one = [_random_walk(rng, 8, 1)]
    two = [_random_walk(rng, 8, 1) for _ in range(2)]
    with pytest.raises(ValueError, match="at least 2"):
        signature_mmd(one, two)
    with pytest.raises(ValueError, match="channel"):
        signature_mmd(two, [_random_walk(rng, 8, 2) for _ in range(2)])


# ---------------------------------------------------------------------------
# signature_mmd_test
# ---------------------------------------------------------------------------


def test_mmd_test_rejects_different_and_keeps_null() -> None:
    rng = np.random.default_rng(61)
    gbm = _gbm_batch(rng, 14, 40, 0.9)
    ou = _ou_batch(rng, 14, 40, 8.0, 0.6)
    alt = signature_mmd_test(
        gbm, ou, seed=101, n_permutations=99, kernel="truncated", order=3, sigma=2.0
    )
    assert alt["reject_5pct"]
    assert 0.0 < alt["p_value"] <= 1.0
    gbm_b = _gbm_batch(rng, 14, 40, 0.9)
    null = signature_mmd_test(
        gbm, gbm_b, seed=102, n_permutations=99, kernel="truncated", order=3, sigma=2.0
    )
    assert not null["reject_5pct"]
    assert null["p_value"] > 0.05


def test_mmd_test_deterministic_seed_and_generator() -> None:
    rng = np.random.default_rng(67)
    a = _gbm_batch(rng, 8, 20, 0.8)
    b = _gbm_batch(rng, 8, 20, 0.8)
    r1 = signature_mmd_test(a, b, seed=5, n_permutations=49, kernel="truncated", order=2)
    r2 = signature_mmd_test(a, b, seed=5, n_permutations=49, kernel="truncated", order=2)
    assert r1 == r2
    gen = np.random.default_rng(5)
    r3 = signature_mmd_test(a, b, seed=gen, n_permutations=49, kernel="truncated", order=2)
    assert r3["p_value"] == r1["p_value"]


def test_mmd_test_permutation_count_and_p_floor() -> None:
    rng = np.random.default_rng(71)
    a = _gbm_batch(rng, 6, 16, 0.8)
    b = _gbm_batch(rng, 6, 16, 0.8)
    res = signature_mmd_test(a, b, seed=3, n_permutations=9, kernel="truncated", order=2)
    assert res["n_permutations"] == 9
    assert res["p_value"] >= 1.0 / 10.0  # randomized-test floor


def test_mmd_test_fail_closed_inputs() -> None:
    rng = np.random.default_rng(73)
    a = _gbm_batch(rng, 4, 12, 0.8)
    b = _gbm_batch(rng, 4, 12, 0.8)
    with pytest.raises(ValueError, match="n_permutations"):
        signature_mmd_test(a, b, seed=1, n_permutations=0)
    with pytest.raises(ValueError, match="seed"):
        signature_mmd_test(a, b, seed="abc")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        signature_mmd_test(a, b, seed=True, n_permutations=5)


# ---------------------------------------------------------------------------
# bench_signature_features — SYNTHETIC bench contract
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def bench_result() -> dict[str, float]:
    return bench_signature_features(BENCH_SEED)


def test_bench_keys_are_synthetic_and_finite(bench_result: dict[str, float]) -> None:
    assert bench_result, "bench returned no keys"
    for key, value in bench_result.items():
        assert key.startswith("synthetic_"), key
        assert math.isfinite(value), key
        assert isinstance(value, float)


def test_bench_deterministic(bench_result: dict[str, float]) -> None:
    assert bench_signature_features(BENCH_SEED) == bench_result


def test_bench_no_forbidden_metric_names(bench_result: dict[str, float]) -> None:
    forbidden = ("sharpe", "sortino", "calmar", "pnl", "nav")
    for key in bench_result:
        assert not any(tok in key.lower() for tok in forbidden)


def test_bench_mmd_power_and_false_positive(bench_result: dict[str, float]) -> None:
    assert bench_result["synthetic_mmd_power_gbm_vs_ou"] >= 0.8
    # 6 seeded replicates under the null; binomial noise around alpha=0.05
    # tolerates at most one spurious rejection here.
    assert bench_result["synthetic_mmd_fp_gbm_vs_gbm"] <= 0.25


def test_bench_vol_shift_and_pde_contrast(bench_result: dict[str, float]) -> None:
    # partial power at a near-boundary sigma contrast (0.4 vs 1.6): must
    # beat chance comfortably but need not be perfect.
    assert bench_result["synthetic_mmd_power_volshift"] >= 0.5
    assert bench_result["synthetic_pde_mmd_reject_gbm_vs_ou"] == 1.0
    assert bench_result["synthetic_pde_mmd_pval_gbm_vs_gbm"] > 0.05


def test_bench_pde_convergence_and_gram_sanity(bench_result: dict[str, float]) -> None:
    assert (
        bench_result["synthetic_pde_refine_gap_l12"] < bench_result["synthetic_pde_refine_gap_l01"]
    )
    assert bench_result["synthetic_pde_vs_truncated_abs_gap"] < 5e-4
    assert bench_result["synthetic_gram_symmetry_err"] < 1e-9
    # PDE Gram is PSD up to discretization error: small negatives only.
    assert bench_result["synthetic_gram_min_eig"] > -0.1
    assert bench_result["synthetic_gram_diag_min"] > 0.0


def test_bench_logsig_drift_regression(bench_result: dict[str, float]) -> None:
    # level-1 displacement dominates drift estimation; R2 ~0.75-0.85 here
    assert bench_result["synthetic_logsig_drift_r2_test"] > 0.5
    assert bench_result["synthetic_logsig_drift_r2_test"] < 1.0
    assert bench_result["synthetic_logsig_feature_dim"] == 5.0


def test_bench_rejects_bad_seed() -> None:
    with pytest.raises(ValueError, match="seed"):
        bench_signature_features(True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="seed"):
        bench_signature_features("x")  # type: ignore[arg-type]
