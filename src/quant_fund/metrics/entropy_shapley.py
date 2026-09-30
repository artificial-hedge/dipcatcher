"""Entropy-Shapley hierarchy for multivariate predictive uncertainty attribution.

Implements the three-level hierarchy from Koenen, Battistin, Van den Abeele &
Jullum (2026, arXiv:2609.35217) "A Hierarchy of Entropy-Shapley Games for
Multivariate Predictive Uncertainty" (submitted 28 Sep 2026).

The hierarchy decomposes how input features drive the predictive uncertainty
of multivariate outputs (e.g. multi-horizon forecasts) through three
entropy-based Shapley games:

  (1) Marginal game  — per-output-component entropy Shapley values φ^(t)_j
  (2) Joint game      — full joint entropy Shapley values φ^joint_j
  (3) Cross-component — the gap Δ_j = Σ_t φ^(t)_j − φ^joint_j, quantifying
      how much each feature shifts the *dependence structure* between output
      components, beyond its effect on per-component marginal uncertainty.

Closed-form expressions are provided for multivariate Gaussian predictive
distributions.  A permutation-based estimator (Strobl & Lantz 2007-style) is
available for many-feature regimes; exact brute-force Shapley is the default
for d ≤ 8 features.

All functions are local (instance-wise) attributions and require only
numpy/scipy — no new dependencies.  Synthetic-only; no market evidence.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]
CovFn = Callable[[Array], Array]
"""Covariance oracle: ``cov_fn(X) -> (N, T, T)`` covariance matrices for N input rows,
T output components.  Each (T, T) slice must be symmetric positive-definite."""


# ---------------------------------------------------------------------------
# Closed-form Gaussian entropy utilities
# ---------------------------------------------------------------------------


def _gaussian_joint_entropy(cov: Array) -> Array:
    """H(Y) = ½ log((2πe)^T × det(Σ)) for each (T, T) covariance slice.

    Args:
        cov: Shape ``(N, T, T)`` — batch of covariance matrices.

    Returns:
        Shape ``(N,)`` — joint differential entropy in nats for each row.

    Raises:
        ValueError: If any covariance matrix is singular (det ≤ 0) or the
                    batch contains non-positive-definite slices.
    """
    cov = np.asarray(cov, dtype=np.float64)
    if cov.ndim != 3 or cov.shape[1] != cov.shape[2]:
        raise ValueError(f"cov must be (N, T, T), got {cov.shape}")
    T = cov.shape[1]
    _, logdet = np.linalg.slogdet(cov)
    if np.any(logdet <= -np.inf) or not np.all(np.isfinite(logdet)):
        raise ValueError("singular or non-positive-definite covariance matrix")
    return np.asarray(0.5 * (T * np.log(2.0 * np.pi * np.e) + logdet), dtype=np.float64)


def _gaussian_marginal_entropy(cov: Array, t: int) -> Array:
    """H(Y_t) = ½ log(2πe × Σ_tt) for output component *t*.

    Args:
        cov: Shape ``(N, T, T)``.
        t:  Zero-based component index.

    Returns:
        Shape ``(N,)`` — marginal entropy in nats.
    """
    var = np.diagonal(cov, axis1=1, axis2=2)[:, t]
    if np.any(var <= 0.0) or not np.all(np.isfinite(var)):
        raise ValueError(f"non-positive or non-finite marginal variance at component t={t}")
    return 0.5 * np.log(2.0 * np.pi * np.e * var)


def _gaussian_sequential_entropy(cov: Array, t: int) -> Array:
    r"""H(Y_t | Y_<t) = ½ log(2πe × σ^2_{t|<t}) via the Schur complement.

    The conditional variance is
      σ^2_{t|<t} = Σ_{tt} − Σ_{t,<t} Σ_{<t,<t}^{-1} Σ_{<t,t}.

    Args:
        cov: Shape ``(N, T, T)``.
        t:  Zero-based component index.

    Returns:
        Shape ``(N,)`` — sequential conditional entropy in nats.
    """
    if t == 0:
        return _gaussian_marginal_entropy(cov, 0)
    _N = cov.shape[0]
    T = cov.shape[1]
    if t < 0 or t >= T:
        raise ValueError(f"t={t} out of range [0, {T})")
    # Extract leading (t+1) × (t+1) submatrices in batch
    sub = cov[:, : t + 1, : t + 1]  # (N, t+1, t+1)
    # Schur complement of leading (t) block, target is last row/col
    sigma_tt = sub[:, t, t]
    sigma_t_lt = sub[:, t, :t]  # (N, t)
    sigma_lt_lt = sub[:, :t, :t]  # (N, t, t)
    sigma_lt_t = sub[:, :t, t]  # (N, t)

    # Solve Σ_{<t,<t} * x = Σ_{<t,t} for each sample → x has shape (N, t).
    # numpy ≥2.0 changed batching behaviour when b.ndim == a.ndim − 1
    # (b is then treated as a stack of (M, K) matrices, not vectors).
    # We reshape b to (N, t, 1) so that b.ndim == a.ndim and the gufunc
    # signature (m,m),(m,n)→(m,n) is satisfied across all numpy versions.
    x = np.linalg.solve(sigma_lt_lt, sigma_lt_t[..., np.newaxis])  # (N, t, 1)
    quad = np.sum(sigma_t_lt * x[..., 0], axis=1)  # (N,)
    cond_var = sigma_tt - quad
    if np.any(cond_var <= 0.0) or not np.all(np.isfinite(cond_var)):
        raise ValueError(f"non-positive conditional variance at t={t}")
    return 0.5 * np.log(2.0 * np.pi * np.e * cond_var)


# ---------------------------------------------------------------------------
# Imputation helpers
# ---------------------------------------------------------------------------


def _impute_coalition(
    x_instance: Array,
    background: Array,
    coalition: frozenset[int],
    rng: np.random.Generator | None = None,
) -> Array:
    """Impute out-of-coalition features from background.

    For coalition S ⊆ [p], each background row contributes one imputed input
    where features in S keep the instance value and features in S̄ are drawn
    from that background row.

    Args:
        x_instance: Shape ``(p,)`` — the instance to explain.
        background: Shape ``(n_bg, p)`` — background reference distribution.
        coalition:  Set of feature indices in the coalition.
        rng:        If given, shuffles background rows before imputing
                    (provides a fresh bootstrap sample per call).

    Returns:
        Shape ``(n_bg, p)`` — imputed feature matrix.
    """
    p = x_instance.shape[0]
    if background.shape[1] != p:
        raise ValueError(
            f"feature dimension mismatch: instance has {p}, background has {background.shape[1]}"
        )
    n_bg = background.shape[0]
    if rng is not None:
        idx = rng.choice(n_bg, size=n_bg, replace=False)
        bg = background[idx]
    else:
        bg = background
    imputed = bg.copy()
    for j in coalition:
        imputed[:, j] = x_instance[j]
    return np.asarray(imputed, dtype=np.float64)


# ---------------------------------------------------------------------------
# Shapley computation engines
# ---------------------------------------------------------------------------


def _shapley_exact(value_fn: Callable[[frozenset[int]], float], p: int) -> Array:
    """Exact Shapley values via enumeration of all 2^p coalitions.

    Uses precomputed combinatorial weights:
      weight(|S|) = |S|! (p - |S| - 1)! / p!

    Args:
        value_fn: Maps a coalition (frozenset of indices) to its payoff.
        p:        Number of features (players).

    Returns:
        Shape ``(p,)`` — Shapley values (efficiency holds: sum = ν([p]) − ν(∅)).
    """
    # Precompute weights for each coalition size
    weights = np.empty(p + 1)
    fact = math.factorial
    for s in range(p + 1):
        if s == p:
            # Term for S = [p]\{j} has |S| = p-1: unused in the sum for φ_j
            # but we never evaluate ν([p]) − ν([p]) for marginal contribution.
            # We handle s = p-1 explicitly.
            continue
        weights[s] = fact(s) * fact(p - s - 1) / fact(p)

    phi = np.zeros(p)

    # Precompute ν for all coalitions
    cache: dict[frozenset[int], float] = {}
    for mask in range(1 << p):
        S = frozenset({j for j in range(p) if mask & (1 << j)})
        cache[S] = value_fn(S)

    for j in range(p):
        for mask in range(1 << p):
            S_set = {i for i in range(p) if mask & (1 << i)}
            if j in S_set:
                continue
            S = frozenset(S_set)
            s = len(S)
            S_plus = frozenset(S_set | {j})
            marginal = cache[S_plus] - cache[S]
            phi[j] += weights[s] * marginal

    return phi


def _shapley_permutation(
    value_fn: Callable[[frozenset[int]], float],
    p: int,
    *,
    n_permutations: int = 200,
    seed: int = 42,
) -> Array:
    """Permutation-based Shapley estimator following Strobl & Lantz (2007).

    For each random feature ordering we walk from ∅ to [p], recording
    the marginal contribution when each feature enters.  Shapley values are
    the average of these contributions across orderings.

    Args:
        value_fn:       Maps a coalition to its payoff.
        p:              Number of features.
        n_permutations: Monte-Carlo orderings (default 200).
        seed:           Deterministic seed.

    Returns:
        Shape ``(p,)`` — approximate Shapley values.
    """
    rng = np.random.default_rng(seed)
    phi = np.zeros(p)

    # Cache ν evaluations: for n_permutations * (p+1) ≈ n_permutations * p
    # distinct coalitions, but many coalitions may repeat across permutations.
    # We cache for efficiency.
    cache: dict[frozenset[int], float] = {}

    def _cached_v(S: frozenset[int]) -> float:
        if S not in cache:
            cache[S] = value_fn(S)
        return cache[S]

    for _ in range(n_permutations):
        order = rng.permutation(p)
        # Start with empty coalition
        S: set[int] = set()
        prev = _cached_v(frozenset(S))
        for idx in order:
            S.add(int(idx))
            curr = _cached_v(frozenset(S))
            phi[idx] += curr - prev
            prev = curr

    return phi / n_permutations


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------


class _AttributionTriplet(NamedTuple):
    """Per-feature attribution triplet for the entropy-Shapley hierarchy."""

    marginal: dict[str, list[float]]
    """Per-output-component φ^(t)_j for each feature j → list of T Shapley values."""

    joint: dict[str, float]
    """Joint φ^joint_j for each feature j → single scalar."""

    cross_component: dict[str, float]
    """Cross-component Δ_j = Σ_t φ^(t)_j − φ^joint_j."""


def entropy_shapley_marginal(
    cov_fn: CovFn,
    x_instance: Array,
    background: Array,
    *,
    feature_names: list[str] | None = None,
    seed: int = 42,
    exact: bool | None = None,
    n_permutations: int = 200,
    max_background: int | None = 256,
) -> dict[str, list[float]]:
    """Per-output-component entropy Shapley values (Level 1 / marginal game).

    For each output component ``t`` and each input feature ``j``, compute the
    Shapley contribution φ^(t)_j of feature ``j`` to the marginal entropy
    H(Y_t | X = x).  This answers: "how much does each feature contribute to
    the predictive uncertainty of component t, ignoring dependencies between
    components?"

    Args:
        cov_fn:          ``cov_fn(X) -> (N, T, T)`` — returns per-sample
                         covariance matrices for the multivariate Gaussian
                         predictive distribution.
        x_instance:      Shape ``(p,)`` — the instance to explain.
        background:      Shape ``(n_bg, p)`` — background reference rows
                         for feature imputation (marginal distribution).
        feature_names:   Optional labels; defaults to ``["f0", "f1", ...]``.
        seed:            Deterministic seed for permutation-based estimation
                         and background shuffling.
        exact:           If ``True``, brute-force enumeration over all 2^p
                         coalitions.  If ``None`` (default), uses exact for
                         ``p ≤ 8``, permutation otherwise.
        n_permutations:  Number of random orderings for the permutation
                         estimator (only used when exact=False).
        max_background:  Cap on background rows (random subsample).  If
                         ``None``, all rows are used.

    Returns:
        ``{feature_name: [φ_t1, φ_t2, ..., φ_tT]}`` — per-component Shapley
        values in nats, one list per feature.

    Raises:
        ValueError: On singular covariance, non-finite entropies, or
                    dimension mismatches.
    """
    # ---- validate inputs ----
    x = np.asarray(x_instance, dtype=np.float64).ravel()
    p = x.shape[0]
    bg = np.asarray(background, dtype=np.float64)
    if bg.ndim != 2 or bg.shape[1] != p:
        raise ValueError(f"background shape {bg.shape} incompatible with instance dim {p}")
    if p < 1:
        raise ValueError("at least one feature required")
    if not np.isfinite(x).all() or not np.isfinite(bg).all():
        raise ValueError("x_instance and background must be finite")

    # Determine T from a test call
    test_cov = np.asarray(cov_fn(x.reshape(1, -1)), dtype=np.float64)
    if test_cov.ndim != 3 or test_cov.shape[1] != test_cov.shape[2]:
        raise ValueError(f"cov_fn must return (N, T, T), got {test_cov.shape}")
    T = test_cov.shape[1]
    if T < 1:
        raise ValueError("at least one output component required")

    names = _resolve_feature_names(feature_names, p)
    use_exact = (p <= 8) if exact is None else exact

    # Subsample background
    bg_use = _subsample_background(bg, max_background, seed)

    # ---- build value function for each component ----
    results: dict[str, list[float]] = {name: [0.0] * T for name in names}

    for t in range(T):
        # Freeze t via default-arg trick to avoid B023 late-binding in lambdas.
        def _v_fn(S: frozenset[int], _t: int = t) -> float:
            return _evaluate_coalition_entropy(bg_use, x, S, cov_fn, _t, seed)

        if use_exact:
            phi_t = _shapley_exact(_v_fn, p)
        else:
            phi_t = _shapley_permutation(_v_fn, p, n_permutations=n_permutations, seed=seed + t)

        for j in range(p):
            results[names[j]][t] = float(phi_t[j])

    return results


def entropy_shapley_joint(
    cov_fn: CovFn,
    x_instance: Array,
    background: Array,
    *,
    feature_names: list[str] | None = None,
    seed: int = 42,
    exact: bool | None = None,
    n_permutations: int = 200,
    max_background: int | None = 256,
) -> dict[str, float]:
    """Joint entropy Shapley values (Level 3 / joint game).

    Computes φ^joint_j — the Shapley contribution of each feature to the
    full joint entropy H(Y₁, ..., Y_T | X = x).  This captures both marginal
    and cross-component dependence effects simultaneously.

    For Gaussian predictive distributions the joint entropy is
    H = ½ log((2πe)^T × det(Σ(x))).

    Args:
        cov_fn:          See :func:`entropy_shapley_marginal`.
        x_instance:      Shape ``(p,)``.
        background:      Shape ``(n_bg, p)``.
        feature_names:   See :func:`entropy_shapley_marginal`.
        seed:            Deterministic seed.
        exact:           Exact for p ≤ 8, permutation otherwise.
        n_permutations:  Permutation count (approx mode).
        max_background:  Background cap.

    Returns:
        ``{feature_name: φ^joint}`` — Shapley values in nats.

    Raises:
        ValueError: On singular covariance, non-finite entropies, or
                    dimension mismatches.
    """
    x = np.asarray(x_instance, dtype=np.float64).ravel()
    p = x.shape[0]
    bg = np.asarray(background, dtype=np.float64)
    if bg.ndim != 2 or bg.shape[1] != p:
        raise ValueError(f"background shape {bg.shape} incompatible with instance dim {p}")
    if p < 1:
        raise ValueError("at least one feature required")
    if not np.isfinite(x).all() or not np.isfinite(bg).all():
        raise ValueError("x_instance and background must be finite")

    test_cov = np.asarray(cov_fn(x.reshape(1, -1)), dtype=np.float64)
    if test_cov.ndim != 3 or test_cov.shape[1] != test_cov.shape[2]:
        raise ValueError(f"cov_fn must return (N, T, T), got {test_cov.shape}")

    names = _resolve_feature_names(feature_names, p)
    use_exact = (p <= 8) if exact is None else exact
    bg_use = _subsample_background(bg, max_background, seed)

    def v_joint(S: frozenset[int]) -> float:
        imputed = _impute_coalition(x, bg_use, S)
        covs = np.asarray(cov_fn(imputed), dtype=np.float64)
        if covs.shape[0] != bg_use.shape[0]:
            raise ValueError(f"cov_fn returned {covs.shape[0]} rows; expected {bg_use.shape[0]}")
        entropies = _gaussian_joint_entropy(covs)
        return float(np.mean(entropies))

    if use_exact:
        phi = _shapley_exact(v_joint, p)
    else:
        phi = _shapley_permutation(v_joint, p, n_permutations=n_permutations, seed=seed)

    return {names[j]: float(phi[j]) for j in range(p)}


def cross_component_attribution(
    marginal: dict[str, list[float]],
    joint: dict[str, float],
    *,
    feature_names: list[str] | None = None,
) -> dict[str, float]:
    """Cross-component attribution Δ_j = Σ_t φ^(t)_j − φ^joint_j.

    Prop. 2 of Koenen et al. (2026): the Shapley values of the total
    correlation (TC) game decompose the deviation of the sum of marginal
    entropy contributions from the joint entropy contribution.

    A large positive Δ_j means feature j increases the dependence between
    output components (making them *more* correlated than the marginals
    suggests).  A large negative Δ_j means it reduces dependence.
    Features that only shift the mean of the multivariate Gaussian
    receive Δ_j ≈ 0.

    Args:
        marginal: Output of :func:`entropy_shapley_marginal`.
        joint:    Output of :func:`entropy_shapley_joint`.
        feature_names: If given, subsets/reorders output (must be a subset
                       of keys in ``marginal``/``joint``).

    Returns:
        ``{feature_name: Δ_j}`` — cross-component attribution in nats.
    """
    keys = list(marginal.keys())
    if feature_names is not None:
        keys = list(feature_names)
        missing_m = [k for k in keys if k not in marginal]
        missing_j = [k for k in keys if k not in joint]
        if missing_m:
            raise ValueError(f"features {missing_m} not in marginal dict")
        if missing_j:
            raise ValueError(f"features {missing_j} not in joint dict")

    result: dict[str, float] = {}
    for name in keys:
        sum_marginal = sum(marginal[name])
        result[name] = sum_marginal - joint[name]
    return result


def entropy_shapley_triplet(
    cov_fn: CovFn,
    x_instance: Array,
    background: Array,
    *,
    feature_names: list[str] | None = None,
    seed: int = 42,
    exact: bool | None = None,
    n_permutations: int = 200,
    max_background: int | None = 256,
) -> _AttributionTriplet:
    """Compute the full attribution triplet (marginal, joint, cross-component).

    Convenience wrapper that runs marginal and joint Shapley decompositions
    and derives the cross-component attribution gap.

    Returns:
        :class:`_AttributionTriplet` with ``.marginal``, ``.joint``, and
        ``.cross_component`` dictionaries.
    """
    marg = entropy_shapley_marginal(
        cov_fn,
        x_instance,
        background,
        feature_names=feature_names,
        seed=seed,
        exact=exact,
        n_permutations=n_permutations,
        max_background=max_background,
    )
    joint = entropy_shapley_joint(
        cov_fn,
        x_instance,
        background,
        feature_names=feature_names,
        seed=seed,
        exact=exact,
        n_permutations=n_permutations,
        max_background=max_background,
    )
    cross = cross_component_attribution(marg, joint)
    return _AttributionTriplet(marginal=marg, joint=joint, cross_component=cross)


# ---------------------------------------------------------------------------
# Synthetic DGP builder (for validation)
# ---------------------------------------------------------------------------


def build_synthetic_cov_fn(
    *,
    T: int = 3,
    seed: int = 42,
) -> tuple[CovFn, dict[str, int], Array]:
    """Build a synthetic multi-output Gaussian model with known feature roles.

    The DGP produces T-horizon multivariate Gaussian forecasts p(Y | x)
    = N(μ(x), Σ(x)) where:

    - Feature 0 ("mean_shift"):       shifts mean only → zero entropy attribution
    - Feature 1 ("variance"):         scales marginal variances (time-increasing)
    - Feature 2 ("var_corr"):         scales both variances and correlation
    - Feature 3 ("corr_only"):        shifts AR(1) correlation without affecting
                                      any marginal variance → invisible to Level 1,
                                      dominant in cross-component attribution.

    The covariance follows an AR(1) structure:
      Σ_ts(x) = σ_t(x) · σ_s(x) · ρ(x)^{|t−s|}

    with
      σ²_t(x) = exp(x₂ · t / T + 0.3 · x₃)   [base + time slope]
      ρ(x)    = tanh(x₃ + x₄)                  [correlation in (−1, 1)]

    Args:
        T:    Number of output components (horizons).
        seed: Deterministic seed for reference instance generation.

    Returns:
        A tuple ``(cov_fn, feature_map, x_ref)`` where:
        - ``cov_fn`` is the callable (N, p) → (N, T, T),
        - ``feature_map`` maps name → index (int),
        - ``x_ref`` is a reference instance with strong correlation.
    """

    def cov_fn(X: Array) -> Array:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != 4:
            raise ValueError("synthetic DGP expects 4 features")

        x2 = X[:, 1]  # variance
        x3 = X[:, 2]  # var + corr
        x4 = X[:, 3]  # corr only

        N = X.shape[0]
        t_arr = np.arange(1, T + 1, dtype=np.float64)  # 1..T

        # Marginal log-variances: depends on x2 (time slope) and x3
        log_var = x2[:, None] * t_arr[None, :] / T + 0.3 * x3[:, None]  # (N, T)
        sigma_t = np.exp(0.5 * log_var)  # (N, T)

        # Correlation ρ ∈ (−1, 1): depends on x3 and x4
        rho = np.tanh(x3 + x4)  # (N,)

        # Build covariance: Σ_ts = σ_t · σ_s · ρ^{|t−s|}
        cov = np.empty((N, T, T), dtype=np.float64)
        for i in range(T):
            for j in range(T):
                dist = abs(i - j)
                cov[:, i, j] = sigma_t[:, i] * sigma_t[:, j] * (rho**dist)
        return cov

    feat_map = {
        "mean_shift": 0,
        "variance": 1,
        "var_corr": 2,
        "corr_only": 3,
    }

    # Reference instance with strong positive correlation
    x_ref = np.array([1.5, 0.4, 1.8, 2.0], dtype=np.float64)

    return cov_fn, feat_map, x_ref


def build_synthetic_background(n: int = 500, *, seed: int = 123) -> Array:
    """Generate background instances for the synthetic DGP.

    Features are drawn i.i.d. ~ N(0, 1), clipped to [-3, 3] for stability.

    Args:
        n:    Number of background rows.
        seed: Deterministic seed.

    Returns:
        Shape ``(n, 4)`` — background feature matrix.
    """
    rng = np.random.default_rng(seed)
    bg = rng.normal(0.0, 1.0, size=(n, 4))
    return np.clip(bg, -3.0, 3.0).astype(np.float64)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_feature_names(names: list[str] | None, p: int) -> list[str]:
    if names is None:
        return [f"f{j}" for j in range(p)]
    if len(names) != p:
        raise ValueError(f"feature_names length {len(names)} != {p}")
    if len(set(names)) != len(names):
        raise ValueError("feature_names must be unique")
    return [str(n) for n in names]


def _subsample_background(bg: Array, max_rows: int | None, seed: int) -> Array:
    if max_rows is not None and max_rows < bg.shape[0]:
        rng = np.random.default_rng(seed)
        idx = rng.choice(bg.shape[0], size=max_rows, replace=False)
        return bg[idx]
    return bg


def _evaluate_coalition_entropy(
    bg: Array,
    x: Array,
    S: frozenset[int],
    cov_fn: CovFn,
    component_t: int,
    seed: int,
) -> float:
    """Evaluate ν(S) = E[H(Y_t | X = (x_S, X_S̄))] for the marginal game."""
    imputed = _impute_coalition(x, bg, S)
    covs = np.asarray(cov_fn(imputed), dtype=np.float64)
    if covs.shape[0] != bg.shape[0]:
        raise ValueError(f"cov_fn returned {covs.shape[0]} rows; expected {bg.shape[0]}")
    entropies = _gaussian_marginal_entropy(covs, component_t)
    return float(np.mean(entropies))
