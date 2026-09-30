"""Synthetic validation of the entropy-Shapley hierarchy (Koenen et al. 2026).

All tests are SYNTHETIC — correctness tests using controlled data-generating
processes, never market evidence.  Determinism is pinned via seeds.
"""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.metrics.entropy_shapley import (
    _gaussian_joint_entropy,
    _gaussian_marginal_entropy,
    _gaussian_sequential_entropy,
    _impute_coalition,
    _shapley_exact,
    _shapley_permutation,
    _subsample_background,
    build_synthetic_background,
    build_synthetic_cov_fn,
    cross_component_attribution,
    entropy_shapley_joint,
    entropy_shapley_marginal,
    entropy_shapley_triplet,
)

# ---------------------------------------------------------------------------
# Gaussian entropy closed forms
# ---------------------------------------------------------------------------


def test_gaussian_joint_entropy_identity() -> None:
    """Joint entropy of independent standard normals = T * ½log(2πe)."""
    T = 3
    ident_cov = np.eye(T, dtype=np.float64)[np.newaxis, :, :]  # (1, T, T)
    h = _gaussian_joint_entropy(ident_cov)[0]
    expected = T * 0.5 * np.log(2.0 * np.pi * np.e)
    assert h == pytest.approx(expected)

    # Repeat for batch of 5
    batch = np.tile(ident_cov, (5, 1, 1))
    h_batch = _gaussian_joint_entropy(batch)
    assert h_batch.shape == (5,)
    assert np.allclose(h_batch, expected)


def test_gaussian_joint_entropy_diagonal() -> None:
    """Joint entropy of diagonal=Σ_i H(Y_i) for independent components."""
    cov = np.array([[[2.0, 0.0], [0.0, 3.0]]], dtype=np.float64)
    h_joint = _gaussian_joint_entropy(cov)[0]
    h1 = 0.5 * np.log(2.0 * np.pi * np.e * 2.0)
    h2 = 0.5 * np.log(2.0 * np.pi * np.e * 3.0)
    assert h_joint == pytest.approx(h1 + h2)


def test_gaussian_marginal_entropy() -> None:
    """Marginal entropy depends only on Σ_tt."""
    cov = np.array([[[4.0, 1.0], [1.0, 9.0]]], dtype=np.float64)
    h0 = _gaussian_marginal_entropy(cov, 0)[0]
    h1 = _gaussian_marginal_entropy(cov, 1)[0]
    assert h0 == pytest.approx(0.5 * np.log(2.0 * np.pi * np.e * 4.0))
    assert h1 == pytest.approx(0.5 * np.log(2.0 * np.pi * np.e * 9.0))
    # Correlation does not affect marginal entropy
    cov_corr = np.array([[[4.0, 3.5], [3.5, 9.0]]], dtype=np.float64)
    assert _gaussian_marginal_entropy(cov_corr, 0)[0] == pytest.approx(h0)


def test_gaussian_sequential_entropy_schur() -> None:
    """Sequential entropy via Schur complement matches formula."""
    # 2D Gaussian with known conditional variance
    var1, var2, rho = 4.0, 9.0, 0.75
    cov12 = rho * np.sqrt(var1 * var2)
    cov = np.array([[[var1, cov12], [cov12, var2]]], dtype=np.float64)

    # t=0: same as marginal
    h0_seq = _gaussian_sequential_entropy(cov, 0)[0]
    h0_marg = _gaussian_marginal_entropy(cov, 0)[0]
    assert h0_seq == pytest.approx(h0_marg)

    # t=1: conditional variance = var2 - cov12^2 / var1
    cond_var = var2 - cov12**2 / var1
    h1_seq = _gaussian_sequential_entropy(cov, 1)[0]
    expected = 0.5 * np.log(2.0 * np.pi * np.e * cond_var)
    assert h1_seq == pytest.approx(expected)


def test_gaussian_entropy_chain_rule() -> None:
    """Chain rule: Σ_t H(Y_t | Y_<t) = H(Y) for Gaussian."""
    T = 3
    rng = np.random.default_rng(42)
    # Build a random positive-definite covariance
    A = rng.normal(0, 1, size=(T, T))
    cov_raw = A @ A.T + 0.5 * np.eye(T)
    cov = cov_raw[np.newaxis, :, :].astype(np.float64)

    h_joint = float(_gaussian_joint_entropy(cov)[0])
    h_seq_sum = sum(_gaussian_sequential_entropy(cov, t)[0].item() for t in range(T))
    assert h_joint == pytest.approx(h_seq_sum)


# ---------------------------------------------------------------------------
# Singular / degenerate edges (fail-closed)
# ---------------------------------------------------------------------------


def test_singular_cov_raises() -> None:
    """Singular covariance → ValueError."""
    cov = np.ones((1, 2, 2), dtype=np.float64)  # rank 1
    with pytest.raises(ValueError):
        _gaussian_joint_entropy(cov)


def test_zero_marginal_variance_raises() -> None:
    """Zero diagonal entry in covariance → ValueError."""
    cov = np.array([[[0.0, 0.0], [0.0, 1.0]]], dtype=np.float64)
    with pytest.raises(ValueError):
        _gaussian_marginal_entropy(cov, 0)


def test_empty_features_raises() -> None:
    """No features → ValueError from Shapley machinery."""
    bg = np.random.default_rng(0).normal(size=(10, 0))
    x = np.array([], dtype=np.float64)

    def bad_cov(_x: np.ndarray) -> np.ndarray:
        return np.ones((10, 1, 1))

    with pytest.raises(ValueError):
        entropy_shapley_marginal(bad_cov, x, bg)


def test_nan_instance_raises() -> None:
    """NaN in x_instance → ValueError."""
    cov_fn, _, _ = build_synthetic_cov_fn()
    bg = build_synthetic_background(50)
    x_nan = np.array([1.0, np.nan, 0.5, -0.3])
    with pytest.raises(ValueError, match="finite"):
        entropy_shapley_marginal(cov_fn, x_nan, bg)


def test_shape_mismatch_raises() -> None:
    """Background with wrong feature count → ValueError."""
    cov_fn, _, x_ref = build_synthetic_cov_fn()
    bg_bad = np.random.default_rng(0).normal(size=(10, 3))  # 3 != 4
    with pytest.raises(ValueError):
        entropy_shapley_marginal(cov_fn, x_ref, bg_bad)


# ---------------------------------------------------------------------------
# Imputation helper
# ---------------------------------------------------------------------------


def test_impute_coalition_preserves_coalition_values() -> None:
    """Coalition features are set to instance values; others from background."""
    x = np.array([10.0, 20.0, 30.0])
    bg = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    imputed = _impute_coalition(x, bg, frozenset({0, 2}))
    assert imputed.shape == (2, 3)
    assert np.allclose(imputed[:, 0], x[0])
    assert np.allclose(imputed[:, 1], bg[:, 1])  # not in coalition
    assert np.allclose(imputed[:, 2], x[2])


def test_impute_coalition_empty() -> None:
    """Empty coalition → all features from background."""
    x = np.array([10.0, 20.0])
    bg = np.array([[1.0, 2.0]])
    imputed = _impute_coalition(x, bg, frozenset())
    assert np.allclose(imputed, bg)


def test_impute_coalition_full() -> None:
    """Full coalition → all features from instance."""
    x = np.array([10.0, 20.0])
    bg = np.array([[1.0, 2.0], [3.0, 4.0]])
    imputed = _impute_coalition(x, bg, frozenset({0, 1}))
    assert np.allclose(imputed, x[np.newaxis, :])


# ---------------------------------------------------------------------------
# Shapley engines
# ---------------------------------------------------------------------------


def test_shapley_exact_efficiency() -> None:
    """Exact Shapley values satisfy efficiency: Σ_j φ_j = ν([p]) − ν(∅)."""

    def v(S: frozenset[int]) -> float:
        # Simple linear game: ν(S) = sum of values in S
        return sum(100.0 if j in S else 0.0 for j in range(4))

    phi = _shapley_exact(v, 4)
    assert phi.sum() == pytest.approx(v(frozenset({0, 1, 2, 3})) - v(frozenset()))
    # For a linear game, each feature's Shapley value equals its direct contribution
    assert np.allclose(phi, 100.0)


def test_shapley_exact_symmetry() -> None:
    """Symmetric players receive equal Shapley values."""

    def v(S: frozenset[int]) -> float:
        return float(len(S)) ** 2

    phi = _shapley_exact(v, 3)
    # All three features are symmetric, so φ values should be equal
    assert np.std(phi) < 1e-10
    full = v(frozenset({0, 1, 2}))
    empty = v(frozenset())
    assert phi.sum() == pytest.approx(full - empty)


def test_shapley_permutation_converges() -> None:
    """Permutation Shapley approximates exact for p=3 with many permutations."""

    def v(S: frozenset[int]) -> float:
        return sum(10.0 * (j + 1) if j in S else 0.0 for j in range(3))

    exact = _shapley_exact(v, 3)
    approx = _shapley_permutation(v, 3, n_permutations=5000, seed=42)
    assert np.allclose(approx, exact, atol=1e-10)


def test_shapley_permutation_deterministic() -> None:
    """Same seed produces identical results."""

    def v(S: frozenset[int]) -> float:
        return float(len(S)) ** 1.5

    phi1 = _shapley_permutation(v, 4, n_permutations=100, seed=42)
    phi2 = _shapley_permutation(v, 4, n_permutations=100, seed=42)
    assert np.allclose(phi1, phi2)


def test_shapley_null_player() -> None:
    """A null player (zero marginal contribution everywhere) gets φ=0."""

    def v(S: frozenset[int]) -> float:
        # Feature 1 never adds value (null player)
        val = 0.0
        if 0 in S:
            val += 5.0
        if 2 in S:
            val -= 3.0
        # Feature 1 contributes nothing regardless
        return val

    phi = _shapley_exact(v, 3)
    assert phi[1] == pytest.approx(0.0, abs=1e-12)
    # Feature 0 should get its full contribution
    assert phi[0] == pytest.approx(5.0)
    assert phi[2] == pytest.approx(-3.0)


# ---------------------------------------------------------------------------
# Cross-component attribution
# ---------------------------------------------------------------------------


def test_cross_component_identity() -> None:
    """Δ_j = Σ_t φ^(t)_j − φ^joint_j holds by construction."""
    marginal = {"a": [1.0, 2.0], "b": [3.0, 4.0]}
    joint = {"a": 2.5, "b": 6.5}
    cross = cross_component_attribution(marginal, joint)
    assert cross["a"] == pytest.approx(3.0 - 2.5)
    assert cross["b"] == pytest.approx(7.0 - 6.5)


def test_cross_component_missing_feature_raises() -> None:
    """Missing feature in either dict → ValueError."""
    marginal = {"a": [1.0, 2.0]}
    joint = {"a": 3.0}
    with pytest.raises(ValueError, match="not in marginal"):
        cross_component_attribution(marginal, joint, feature_names=["b"])


# ---------------------------------------------------------------------------
# Background subsampling
# ---------------------------------------------------------------------------


def test_subsample_background_caps() -> None:
    """max_background limits rows; same seed yields reproducible cap."""
    bg = np.random.default_rng(0).normal(size=(200, 4))
    capped = _subsample_background(bg, 50, seed=42)
    assert capped.shape == (50, 4)

    capped2 = _subsample_background(bg, 50, seed=42)
    assert np.allclose(capped, capped2)


# ---------------------------------------------------------------------------
# Synthetic DGP validation (the paper's key results)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def synthetic_setup() -> tuple:
    """Return (cov_fn, feat_map, x_ref, background) for the synthetic DGP."""
    cov_fn, feat_map, x_ref = build_synthetic_cov_fn(T=3, seed=42)
    bg = build_synthetic_background(500, seed=123)
    return cov_fn, feat_map, x_ref, bg


def test_mean_shift_zero_attribution(synthetic_setup: tuple) -> None:
    """Feature that shifts only the mean gets zero entropy attribution."""
    cov_fn, feat_map, x_ref, bg = synthetic_setup

    marg = entropy_shapley_marginal(cov_fn, x_ref, bg, feature_names=list(feat_map), seed=42)
    joint = entropy_shapley_joint(cov_fn, x_ref, bg, feature_names=list(feat_map), seed=42)

    # mean_shift (index 0) controls μ only → Gaussian entropy is μ-independent
    assert abs(marg["mean_shift"][0]) < 1e-12
    assert abs(marg["mean_shift"][1]) < 1e-12
    assert abs(marg["mean_shift"][2]) < 1e-12
    assert abs(joint["mean_shift"]) < 1e-12


def test_corr_only_invisible_to_marginal(synthetic_setup: tuple) -> None:
    """corr_only feature gets zero Level-1 (marginal) attribution."""
    cov_fn, feat_map, x_ref, bg = synthetic_setup

    marg = entropy_shapley_marginal(cov_fn, x_ref, bg, feature_names=list(feat_map), seed=42)

    # corr_only (index 3) does not affect marginal variances → zero Level 1
    for t in range(3):
        assert abs(marg["corr_only"][t]) < 1e-6, (
            f"corr_only non-zero at t={t}: {marg['corr_only'][t]}"
        )


def test_corr_only_large_cross_component(synthetic_setup: tuple) -> None:
    """corr_only feature has large cross-component attribution (negative)."""
    cov_fn, feat_map, x_ref, bg = synthetic_setup

    marg = entropy_shapley_marginal(cov_fn, x_ref, bg, feature_names=list(feat_map), seed=42)
    joint = entropy_shapley_joint(cov_fn, x_ref, bg, feature_names=list(feat_map), seed=42)
    cross = cross_component_attribution(marg, joint)

    # corr_only should have significant cross-component attribution
    # (it affects correlation → reduces joint entropy relative to sum of marginals)
    cross_corr = cross["corr_only"]
    assert abs(cross_corr) > 0.001, f"cross-component too small: {cross_corr}"

    # corr_only has near-zero marginal sum but significant cross-component
    sum_marg_corr = sum(marg["corr_only"])
    assert abs(sum_marg_corr) < 1e-6
    assert abs(cross_corr) > abs(sum_marg_corr) * 10, (
        f"cross-component should dominate marginal for corr_only: "
        f"cross={cross_corr}, sum_marg={sum_marg_corr}"
    )


def test_chain_rule_decomposition_quality(synthetic_setup: tuple) -> None:
    """Chain-rule decomposes joint into sum-of-marginals − cross-component.

    The identity Σ_t φ^(t)_j − Δ_j = φ^joint_j must hold exactly by the
    definition Δ_j = Σ_t φ^(t)_j − φ^joint_j.  We verify it.
    """
    cov_fn, feat_map, x_ref, bg = synthetic_setup

    marg = entropy_shapley_marginal(cov_fn, x_ref, bg, feature_names=list(feat_map), seed=42)
    joint = entropy_shapley_joint(cov_fn, x_ref, bg, feature_names=list(feat_map), seed=42)
    cross = cross_component_attribution(marg, joint)

    for name in feat_map:
        sum_marg = sum(marg[name])
        # Chain-rule identity: sum_marg − cross = joint
        residual = sum_marg - cross[name] - joint[name]
        assert abs(residual) < 1e-14, f"chain-rule residual for '{name}': {residual:.2e}"


def test_cross_component_ranking(synthetic_setup: tuple) -> None:
    """corr_only should have largest |cross-component| attribution."""
    cov_fn, feat_map, x_ref, bg = synthetic_setup

    marg = entropy_shapley_marginal(cov_fn, x_ref, bg, feature_names=list(feat_map), seed=42)
    joint = entropy_shapley_joint(cov_fn, x_ref, bg, feature_names=list(feat_map), seed=42)
    cross = cross_component_attribution(marg, joint)

    # corr_only should have largest absolute cross-component
    abs_cross = {name: abs(v) for name, v in cross.items()}
    top = max(abs_cross, key=lambda k: abs_cross[k])  # type: ignore[arg-type]
    assert top == "corr_only", (
        f"expected corr_only to dominate cross-component, got {top}. Cross values: {cross}"
    )


def test_variance_features_have_marginal_attribution(synthetic_setup: tuple) -> None:
    """variance and var_corr features get significant Level-1 attribution."""
    cov_fn, feat_map, x_ref, bg = synthetic_setup

    marg = entropy_shapley_marginal(cov_fn, x_ref, bg, feature_names=list(feat_map), seed=42)

    # variance (index 1) affects marginal Σ_tt → non-zero Level 1
    for t in range(3):
        assert abs(marg["variance"][t]) > 1e-4, f"variance near-zero at t={t}"

    # var_corr (index 2) also affects marginals
    for t in range(3):
        assert abs(marg["var_corr"][t]) > 1e-4, f"var_corr near-zero at t={t}"


def test_time_increasing_variance_effect(synthetic_setup: tuple) -> None:
    """variance feature has time-increasing attribution (time slope in DGP)."""
    cov_fn, feat_map, x_ref, bg = synthetic_setup

    marg = entropy_shapley_marginal(cov_fn, x_ref, bg, feature_names=list(feat_map), seed=42)

    # variance's log-variance is proportional to t/T, so later horizons
    # should have larger attribution (same sign)
    v = marg["variance"]
    # Values should all have the same sign (determined by x_ref)
    if v[0] > 0:
        assert v[2] > v[0], f"time-increasing expected: {v}"
    else:
        assert v[2] < v[0], f"time-increasing expected (negative): {v}"


# ---------------------------------------------------------------------------
# Triplet convenience wrapper
# ---------------------------------------------------------------------------


def test_entropy_shapley_triplet_consistent(synthetic_setup: tuple) -> None:
    """Triplet wrapper agrees with individual calls."""
    cov_fn, feat_map, x_ref, bg = synthetic_setup
    names = list(feat_map)

    triplet = entropy_shapley_triplet(cov_fn, x_ref, bg, feature_names=names, seed=42)
    marg = entropy_shapley_marginal(cov_fn, x_ref, bg, feature_names=names, seed=42)
    joint = entropy_shapley_joint(cov_fn, x_ref, bg, feature_names=names, seed=42)
    cross = cross_component_attribution(marg, joint)

    for name in names:
        assert triplet.marginal[name] == pytest.approx(marg[name])
        assert triplet.joint[name] == pytest.approx(joint[name])
        assert triplet.cross_component[name] == pytest.approx(cross[name])


# ---------------------------------------------------------------------------
# Permutation vs exact agreement (small-p)
# ---------------------------------------------------------------------------


def test_permutation_agrees_with_exact_small_p(synthetic_setup: tuple) -> None:
    """Permutation Shapley approximates exact for p=4 with enough perms."""
    cov_fn, feat_map, x_ref, bg = synthetic_setup
    names = list(feat_map)

    marg_exact = entropy_shapley_marginal(
        cov_fn, x_ref, bg, feature_names=names, seed=42, exact=True
    )
    marg_perm = entropy_shapley_marginal(
        cov_fn, x_ref, bg, feature_names=names, seed=42, exact=False, n_permutations=5000
    )

    joint_exact = entropy_shapley_joint(cov_fn, x_ref, bg, feature_names=names, seed=42, exact=True)
    joint_perm = entropy_shapley_joint(
        cov_fn, x_ref, bg, feature_names=names, seed=42, exact=False, n_permutations=5000
    )

    for name in names:
        for t in range(3):
            assert marg_exact[name][t] == pytest.approx(marg_perm[name][t], abs=1e-1), (
                f"{name}[{t}]: exact={marg_exact[name][t]}, perm={marg_perm[name][t]}"
            )
        assert joint_exact[name] == pytest.approx(joint_perm[name], abs=1e-1), (
            f"{name} joint: exact={joint_exact[name]}, perm={joint_perm[name]}"
        )


# ---------------------------------------------------------------------------
# Chain-rule decomposition cross-check (via sequential entropy construction)
# ---------------------------------------------------------------------------


def test_chain_rule_via_sequential() -> None:
    """The sum of sequential (Level-2) Shapley equals the joint (Level-3).

    This tests Prop. 1 (chain-rule linkage) from Koenen et al. (2026):
    φ^joint_j = Σ_t φ^(t|<t)_j.

    We verify this numerically on the synthetic DGP by computing both sides.
    Since Level 2 uses H(Y_t|Y_<t) via the Schur complement, and the chain
    rule guarantees Σ H(Y_t|Y_<t) = H(Y), the Shapley decomposition must
    also satisfy the additivity.
    """
    cov_fn, feat_map, x_ref = build_synthetic_cov_fn(T=3, seed=42)
    bg_small = build_synthetic_background(100, seed=123)
    names = list(feat_map)
    p = len(names)

    # Compute exact Level-3 (joint) Shapley
    joint = entropy_shapley_joint(cov_fn, x_ref, bg_small, feature_names=names, seed=42, exact=True)

    # Compute Level-2 (sequential) Shapley per component
    # We need a value function ν^(t|<t)(S) = E[H(Y_t | Y_<t, X = (x_S, X_S̄))]
    # Using the Gaussian sequential entropy via Schur complement.
    def _sequential_v_fn(S: frozenset[int], t: int) -> float:
        imputed = _impute_coalition(x_ref, bg_small, S)
        covs = np.asarray(cov_fn(imputed), dtype=np.float64)
        entropies = _gaussian_sequential_entropy(covs, t)
        return float(np.mean(entropies))

    # Sum Level-2 Shapley values across components
    sum_level2 = np.zeros(p)
    for t in range(3):
        # Freeze t in closure via default-arg trick
        def _v_fn(S: frozenset[int], _t: int = t) -> float:
            return _sequential_v_fn(S, _t)

        phi_t = _shapley_exact(_v_fn, p)
        sum_level2 += phi_t

    # Prop. 1: sum of Level-2 must equal Level-3
    for j, name in enumerate(names):
        residual = abs(sum_level2[j] - joint[name])
        assert residual < 1e-13, (
            f"Prop.1 violation for '{name}': Level3={joint[name]}, ΣLevel2={sum_level2[j]}, Δ={residual:.2e}"
        )


# ---------------------------------------------------------------------------
# Export format: dict ready for bar-chart / report
# ---------------------------------------------------------------------------


def test_output_dict_ready_for_report(synthetic_setup: tuple) -> None:
    """Attribution triplet exports as plain dicts suitable for plotting."""
    cov_fn, feat_map, x_ref, bg = synthetic_setup
    names = list(feat_map)

    triplet = entropy_shapley_triplet(cov_fn, x_ref, bg, feature_names=names, seed=42)

    # Check types
    assert isinstance(triplet.marginal, dict)
    assert isinstance(triplet.joint, dict)
    assert isinstance(triplet.cross_component, dict)

    for name in names:
        assert isinstance(triplet.marginal[name], list)
        assert len(triplet.marginal[name]) == 3  # T horizons
        assert isinstance(triplet.joint[name], float)
        assert isinstance(triplet.cross_component[name], float)

    # Chain-rule identity: Σ marg − cross = joint for every feature
    for name in names:
        sum_marg = sum(triplet.marginal[name])
        residual = sum_marg - triplet.cross_component[name] - triplet.joint[name]
        assert abs(residual) < 1e-14


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_output_deterministic(synthetic_setup: tuple) -> None:
    """Same inputs produce identical outputs (seeded)."""
    cov_fn, feat_map, x_ref, bg = synthetic_setup
    names = list(feat_map)

    def compute() -> tuple:
        marg = entropy_shapley_marginal(cov_fn, x_ref, bg, feature_names=names, seed=42)
        joint = entropy_shapley_joint(cov_fn, x_ref, bg, feature_names=names, seed=42)
        cross = cross_component_attribution(marg, joint)
        return marg, joint, cross

    m1, j1, c1 = compute()
    m2, j2, c2 = compute()

    for name in names:
        assert m1[name] == pytest.approx(m2[name])
        assert j1[name] == pytest.approx(j2[name])
        assert c1[name] == pytest.approx(c2[name])
