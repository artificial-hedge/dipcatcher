"""Generalized hierarchical conformal prediction (GHCP).

Mallick, Tchetgen Tchetgen, Dobriban and Lee (2026), "Generalized
Hierarchical Conformal Prediction", arXiv:2608.15500 [stat.ME],
https://arxiv.org/abs/2608.15500 (fetched and verified 2026-09-30).
Authors' code: https://github.com/soham-penn/hierarchical_CP. Extends
hierarchical conformal prediction (HCP; Lee, Barber and Willett (2026),
ACM Journal of Data Science, doi:10.1145/3786352) from the first
observation of an unseen group to the (o+1)-th observation of a test
group whose first ``o >= 0`` records are already collected.

Setting: K reference groups with sizes N_1..N_K and a test group with o
initial observations; predict Y_{K+1,o+1}. Validity (paper Theorem 2.1,
Corollary 2.6, Remark 2.2) is finite-sample and distribution-free under

- A1 group-level exchangeability of (U_j, mu_j),
- A2 group-size ignorability: N_{1:K} independent of the group laws
  (load-bearing for the donation step),
- A3 within-group i.i.d. sampling given the group law,
- A4 (no ties) only for the coverage upper bound.

Construction (paper Section 2.1 / Algorithm 1):

1. Selection. Compatible donors ``S = {j : N_j > o}``; restricted pool
   ``S_eta`` keeps the ``m_eta = min(|S|, ceil((1-eta) K))`` smallest
   compatible sizes, cutoff ties broken uniformly at random in a
   permutation-equivariant way (eq. (8), Appendix B.2).
2. Donation. ``J_0 ~ Unif(S_eta)``; the test group receives the
   surrogate size ``N_{K+1} = N_{J_0}``; ``S_cal = S_eta \\ {J_0}``.
   If ``S_eta`` is empty: ``S_cal = {}`` and ``N_{K+1} = o + 1``.
3. Global + local training. ``S_train = [K] \\ S_eta`` trains the global
   predictor (any learner is allowed, Appendix B.1; this module uses a
   numpy ridge fit, or an intercept-only mean without covariates, in
   place of the paper's random forest). Each participating group keeps
   its first ``tau = floor(o/2)`` responses as a local block; the merged
   predictor is ``mu~_j = (1 - lam) mu_glob + lam mean(Y_{j,1..tau})``
   with ``lam = tau / (|S_train| + tau)`` (eq. (3); ``|S_train|`` counts
   groups, per the authors' code README). Score ``s = |Y - mu~_j(X)|``.
4. Calibration measure ``nu_donor`` (eq. (5)): held-out scores of group
   ``j in S_cal`` (positions tau..N_j-1) each carry weight
   ``1 / ((|S_cal|+1) L_j)``, ``L_j = N_j - tau``; the test group's
   held-out scores (positions tau..o-1) each carry
   ``1 / ((|S_cal|+1) L_{K+1})``; the ``N_{K+1} - o`` unobserved test
   slots are one ``+inf`` atom of weight
   ``(N_{K+1} - o) / ((|S_cal|+1) L_{K+1})``. The threshold is
   ``q = Q_{1-alpha}(nu_donor)`` (footnote-2 quantile); ``q = +inf``
   when the level is unattainable, i.e. a trivial (honest) set rather
   than a clipped one. Prediction set ``{y : |y - mu~_{K+1}(x)| <= q}``.

Reductions (pinned as closed-form equalities in the unit tests):

- ``o = 0``: ``nu_donor`` is exactly the Lee et al. (2026) HCP measure
  over ``K_1 = |S_cal|`` calibration groups -- each group total weight
  ``1/(K_1+1)``, placeholder ``1/(K_1+1)`` at ``+inf`` -- so ``q`` is
  the HCP weighted quantile, equivalently the ``(1-alpha)(K_1+1)/K_1``
  quantile among the finite scores.
- ``S = {}`` (no reference group larger than ``o``, e.g. the whole group
  is already observed): exactly within-test-group split conformal with
  ``n_1 = o - floor(o/2)`` calibration scores and surrogate size
  ``o + 1`` (Remark 2.2); ``q`` is the ``ceil((n_1+1)(1-alpha))``-th
  order statistic when attainable and ``+inf`` otherwise.

Implementation notes / deviations, all documented:

- The paper's pool size ``m_eta = min(|S|, ceil((1-eta) K))`` is
  implemented as written; the authors' released code selects one extra
  pool member (``n_sel + 1``) relative to eq. (8). The paper formula is
  authoritative here.
- Group-level features ``U_j`` are not supported; the paper allows
  omitting ``U`` when unavailable. Optional per-observation covariates
  ``X`` enter the ridge global predictor only (the local predictor is
  the group mean, as in the paper).
- ``QUANTILE_TOL = 1e-12`` guards float representation error in the
  weighted-CDF comparison; at exact boundaries (``inf`` weight equal to
  ``alpha``) the mathematical quantile is the largest finite atom.
- Fail-closed: invalid inputs raise ``ValueError``; unattainable levels
  return ``+inf`` thresholds with ``trivial=True`` (never clipped);
  ``MAX_GROUPS`` caps the reference-group count.

Determinism: all randomness (pool tie-breaks, donor draw) flows through
``np.random.default_rng(seed)``; the same seed and data give bitwise
identical results, independent of global numpy RNG state.

SYNTHETIC results from :func:`bench_ghcp` are correctness tests, always
labeled, never market evidence. Proper set scores only (coverage, set
width). No Sharpe.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]

MAX_GROUPS = 100_000
QUANTILE_TOL = 1e-12


@dataclass(frozen=True, eq=False)
class GHCPResult:
    """One GHCP prediction set plus the calibration atoms that produced it.

    Exposing the atoms (scores, weights, ``inf_weight``) lets tests pin the
    reduction equalities in closed form without re-fitting predictors.
    """

    lower: float
    upper: float
    threshold: float
    center: float
    trivial: bool
    donor_index: int | None
    s_pool: tuple[int, ...]
    s_cal: tuple[int, ...]
    s_train: tuple[int, ...]
    surrogate_size: int
    n_initial: int
    tau: int
    lam: float
    alpha: float
    eta: float
    seed: int
    cal_scores: Array
    cal_weights: Array
    test_scores: Array
    test_weight: float
    inf_atoms: int
    inf_weight: float


def weighted_measure_quantile(
    scores: Array, weights: Array, inf_weight: float, alpha: float
) -> float:
    r"""Q_{1-alpha}(nu) for nu = sum_i w_i delta_{s_i} + w_inf delta_{+inf}.

    Paper footnote 2: ``Q_beta(nu) = inf{t : nu((-inf, t]) >= beta}``. Atoms
    are normalized to a probability measure; ``+inf`` is returned when the
    level is unattainable among the finite atoms (``w_inf > alpha`` up to
    ``QUANTILE_TOL``) so callers see an honest trivial set instead of a
    clipped quantile.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    s = np.asarray(scores, dtype=float).ravel()
    w = np.asarray(weights, dtype=float).ravel()
    if s.size != w.size:
        raise ValueError("scores and weights must have the same length")
    if s.size > 0 and not bool(np.all(np.isfinite(s))):
        raise ValueError("scores must be finite; carry +inf via inf_weight")
    if bool(np.any(w < 0.0)) or not np.isfinite(inf_weight) or inf_weight < 0.0:
        raise ValueError("weights must be finite and non-negative")
    total = float(np.sum(w)) + float(inf_weight)
    if total <= 0.0 or not np.isfinite(total):
        return float("inf")
    if s.size == 0:
        return float("inf")
    beta = 1.0 - float(alpha)
    order = np.argsort(s, kind="mergesort")
    cumulative = np.cumsum(w[order]) / total
    idx = int(np.searchsorted(cumulative, beta - QUANTILE_TOL, side="left"))
    if idx >= int(s.size):
        return float("inf")
    return float(s[order][idx])


def donor_pool(
    sizes: Array | NDArray[np.int64], n_initial: int, eta: float, rng: np.random.Generator
) -> NDArray[np.int64]:
    """Restricted donor pool ``S_eta`` (paper eq. (8) and Appendix B.2).

    Keeps the ``m_eta = min(|S|, ceil((1-eta) K))`` compatible groups
    (``N_j > n_initial``) with the smallest sizes; cutoff ties are broken by
    a uniform random subset drawn from ``rng`` (permutation-equivariant).
    ``eta = 0`` returns the unrestricted compatible set ``S``.
    """
    if not 0.0 <= eta < 1.0:
        raise ValueError("eta must be in [0, 1)")
    n = np.asarray(sizes)
    k = int(n.size)
    compatible = np.flatnonzero(n > n_initial)
    if compatible.size == 0:
        return np.empty(0, dtype=np.int64)
    m_eta = min(int(compatible.size), int(np.ceil(round((1.0 - eta) * k, 9))))
    if m_eta <= 0:
        return np.empty(0, dtype=np.int64)
    cutoff = int(np.partition(n[compatible], m_eta - 1)[m_eta - 1])
    below = compatible[n[compatible] < cutoff]
    tied = compatible[n[compatible] == cutoff]
    need = m_eta - int(below.size)
    chosen: NDArray[np.int64]
    if need <= 0:
        chosen = np.empty(0, dtype=np.int64)
    elif need >= int(tied.size):
        chosen = np.asarray(tied, dtype=np.int64)
    else:
        chosen = np.sort(np.asarray(rng.choice(tied, size=need, replace=False), dtype=np.int64))
    return np.sort(np.concatenate([below.astype(np.int64), chosen]))


@dataclass(frozen=True)
class _GlobalFit:
    """Global predictor fit on ``S_train`` only; ``coef is None`` => constant."""

    coef: Array | None
    intercept: float
    x_mean: Array | None

    def predict_rows(self, x: Array | None, n_rows: int) -> Array:
        if self.coef is None or self.x_mean is None:
            return np.full(int(n_rows), float(self.intercept), dtype=float)
        xa = np.asarray(x, dtype=float).reshape(int(n_rows), int(self.x_mean.size))
        out = float(self.intercept) + (xa - self.x_mean) @ self.coef
        return np.asarray(out, dtype=float)

    def predict_one(self, x: Array | None) -> float:
        if self.coef is None or self.x_mean is None:
            return float(self.intercept)
        xa = np.asarray(x, dtype=float).ravel()
        return float(float(self.intercept) + float((xa - self.x_mean) @ self.coef))


def _fit_global(x_rows: Array | None, y_rows: Array, ridge: float) -> _GlobalFit:
    if int(y_rows.size) == 0:
        # Paper footnote 1: S_train empty => mu_glob == 0.
        return _GlobalFit(None, 0.0, None)
    if x_rows is None:
        return _GlobalFit(None, float(np.mean(y_rows)), None)
    x = np.asarray(x_rows, dtype=float)
    x_mean = np.asarray(np.mean(x, axis=0), dtype=float)
    y_mean = float(np.mean(y_rows))
    centered = x - x_mean
    gram = centered.T @ centered + float(ridge) * np.eye(int(x.shape[1]))
    try:
        coef = np.linalg.solve(gram, centered.T @ (np.asarray(y_rows, dtype=float) - y_mean))
    except np.linalg.LinAlgError as exc:
        raise ValueError("global ridge design is singular; increase ridge") from exc
    return _GlobalFit(np.asarray(coef, dtype=float), y_mean, x_mean)


def _as_1d_finite(values: object, name: str) -> Array:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-d")
    if arr.size > 0 and not bool(np.all(np.isfinite(arr))):
        raise ValueError(f"{name} must be finite")
    return arr


def _validate_covariates(
    x_groups: Sequence[Array] | None,
    groups: list[Array],
    n_initial: int,
    x_test_initial: Array | None,
    x_test_target: Array | None,
) -> tuple[list[Array] | None, Array | None, Array | None]:
    if x_groups is None:
        if x_test_initial is not None or x_test_target is not None:
            raise ValueError("x_test_initial/x_test_target require x_groups")
        return None, None, None
    if len(x_groups) != len(groups):
        raise ValueError("x_groups must align with y_groups")
    width = -1
    prepared: list[Array] = []
    for j, (xj, yj) in enumerate(zip(x_groups, groups, strict=True)):
        xa = np.asarray(xj, dtype=float)
        if xa.ndim == 1:
            xa = xa.reshape(-1, 1)
        if xa.ndim != 2 or int(xa.shape[0]) != int(yj.size):
            raise ValueError(f"x_groups[{j}] must have one row per observation")
        if not bool(np.all(np.isfinite(xa))):
            raise ValueError(f"x_groups[{j}] must be finite")
        if width < 0:
            width = int(xa.shape[1])
        elif int(xa.shape[1]) != width:
            raise ValueError("x_groups must share one covariate width")
        prepared.append(xa)
    if x_test_target is None:
        raise ValueError("x_test_target is required when covariates are used")
    x_target = np.asarray(x_test_target, dtype=float).ravel()
    if int(x_target.size) != width:
        raise ValueError("x_test_target width must match x_groups")
    if not bool(np.all(np.isfinite(x_target))):
        raise ValueError("x_test_target must be finite")
    x_test_i: Array | None = None
    if n_initial > 0:
        if x_test_initial is None:
            raise ValueError("x_test_initial is required when covariates are used")
        xt = np.asarray(x_test_initial, dtype=float)
        if xt.ndim == 1:
            xt = xt.reshape(int(n_initial), -1)
        if xt.ndim != 2 or xt.shape != (n_initial, width):
            raise ValueError("x_test_initial must have shape (n_initial, width)")
        if not bool(np.all(np.isfinite(xt))):
            raise ValueError("x_test_initial must be finite")
        x_test_i = xt
    return prepared, x_test_i, x_target


def hcp_predict(
    y_groups: Sequence[Array],
    y_test_initial: Array,
    *,
    x_groups: Sequence[Array] | None = None,
    x_test_initial: Array | None = None,
    x_test_target: Array | None = None,
    alpha: float = 0.10,
    eta: float = 0.5,
    ridge: float = 1e-6,
    seed: int = 0,
) -> GHCPResult:
    """GHCP prediction interval for the next observation of the test group.

    ``y_groups[j]`` are the reference-group responses (ragged sizes allowed);
    ``y_test_initial`` holds the ``o`` test-group responses observed before
    prediction. The target is the response of the next test-group
    observation; with covariates, ``x_test_target`` is its feature row. The
    absolute-residual score gives the interval ``center +/- threshold``;
    ``threshold = +inf`` (``trivial=True``) when the ``1 - alpha`` level is
    unattainable, which is the honest fail-closed output.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if not np.isfinite(ridge) or float(ridge) < 0.0:
        raise ValueError("ridge must be finite and non-negative")
    k = len(y_groups)
    if k > MAX_GROUPS:
        raise ValueError(f"reference group count {k} exceeds MAX_GROUPS={MAX_GROUPS}")
    groups = [_as_1d_finite(g, "y_groups[j]") for j, g in enumerate(y_groups)]
    for j, g in enumerate(groups):
        if int(g.size) == 0:
            raise ValueError(f"reference group {j} is empty")
    y_test = _as_1d_finite(y_test_initial, "y_test_initial")
    o = int(y_test.size)
    if k == 0 and o == 0:
        raise ValueError("hcp_predict needs >=1 reference group or >=1 initial observation")
    xg, x_test_i, x_target = _validate_covariates(
        x_groups, groups, o, x_test_initial, x_test_target
    )

    sizes = np.array([int(g.size) for g in groups], dtype=np.int64)
    rng = np.random.default_rng(seed)
    pool = donor_pool(sizes, o, float(eta), rng)
    s_train = np.setdiff1d(np.arange(k, dtype=np.int64), pool)
    tau = o // 2

    if int(s_train.size) > 0:
        y_tr = np.concatenate([groups[int(j)] for j in s_train])
        x_tr = None if xg is None else np.vstack([xg[int(j)] for j in s_train])
    else:
        y_tr = np.empty(0, dtype=float)
        x_tr = None
    glob = _fit_global(x_tr, y_tr, float(ridge))

    donor: int | None = None
    if int(pool.size) > 0:
        donor = int(rng.choice(pool))
        s_cal = np.setdiff1d(pool, np.asarray([donor], dtype=np.int64))
        surrogate = int(sizes[donor])
    else:
        s_cal = np.empty(0, dtype=np.int64)
        surrogate = o + 1

    lam_denominator = int(s_train.size) + tau
    lam = float(tau) / float(lam_denominator) if lam_denominator > 0 else 0.0
    s_size = int(s_cal.size) + 1

    cal_score_parts: list[Array] = []
    cal_weight_parts: list[Array] = []
    for grp in s_cal:
        jj = int(grp)
        tail = groups[jj][tau:]
        x_tail = None if xg is None else xg[jj][tau:]
        mu_glob_rows = glob.predict_rows(x_tail, int(tail.size))
        mu_loc = float(groups[jj][:tau].mean()) if tau > 0 else 0.0
        mu_tilde = (1.0 - lam) * mu_glob_rows + lam * mu_loc
        cal_score_parts.append(np.abs(tail - mu_tilde))
        cal_weight_parts.append(np.full(int(tail.size), 1.0 / (s_size * (int(sizes[jj]) - tau))))

    l_test = surrogate - tau
    w_test = 1.0 / (s_size * l_test)
    mu_loc_test = float(y_test[:tau].mean()) if tau > 0 else 0.0
    if o > tau:
        x_test_tail = None if x_test_i is None else x_test_i[tau:o]
        mu_glob_test = glob.predict_rows(x_test_tail, o - tau)
        mu_tilde_test = (1.0 - lam) * mu_glob_test + lam * mu_loc_test
        test_scores = np.abs(y_test[tau:o] - mu_tilde_test)
    else:
        test_scores = np.empty(0, dtype=float)
    n_inf = surrogate - o
    inf_weight = float(n_inf) * w_test

    score_parts = [*cal_score_parts]
    weight_parts = [*cal_weight_parts]
    if int(test_scores.size) > 0:
        score_parts.append(test_scores)
        weight_parts.append(np.full(int(test_scores.size), w_test))
    finite = np.concatenate(score_parts) if score_parts else np.empty(0, dtype=float)
    weights = np.concatenate(weight_parts) if weight_parts else np.empty(0, dtype=float)

    threshold = weighted_measure_quantile(finite, weights, inf_weight, alpha)
    trivial = bool(np.isinf(threshold))
    center = float((1.0 - lam) * glob.predict_one(x_target) + lam * mu_loc_test)
    lower = float("-inf") if trivial else float(center - threshold)
    upper = float("inf") if trivial else float(center + threshold)

    return GHCPResult(
        lower=lower,
        upper=upper,
        threshold=float(threshold),
        center=center,
        trivial=trivial,
        donor_index=donor,
        s_pool=tuple(int(v) for v in pool),
        s_cal=tuple(int(v) for v in s_cal),
        s_train=tuple(int(v) for v in s_train),
        surrogate_size=int(surrogate),
        n_initial=o,
        tau=int(tau),
        lam=float(lam),
        alpha=float(alpha),
        eta=float(eta),
        seed=int(seed),
        cal_scores=np.asarray(finite[: sum(int(p.size) for p in cal_score_parts)]),
        cal_weights=np.asarray(weights[: sum(int(p.size) for p in cal_weight_parts)]),
        test_scores=np.asarray(test_scores),
        test_weight=float(w_test),
        inf_atoms=int(n_inf),
        inf_weight=float(inf_weight),
    )


def bench_ghcp(
    n_reps: int = 200,
    alpha: float = 0.10,
    m_values: Sequence[int] = (0, 1, 3, 10),
    seed: int = 11,
    n_groups: int = 20,
    base_size: int = 16,
    gamma: float = 5.0,
    sigma: float = 1.0,
    eta: float = 0.5,
) -> dict[str, float | str]:
    """SYNTHETIC hierarchical-Gaussian correctness bench. Never market evidence.

    Paper Section 3.1 DGP with covariates removed: group effects
    ``B_j ~ N(0, gamma^2)``, within-group noise ``N(0, sigma^2)``, reference
    sizes ``base_size..base_size+K-1`` (distinct, all ``> max(m_values)`` so
    donors exist), one test stream; the scored target sits at stream position
    ``max(m_values)`` for every ``m`` (as in the authors' runners, where the
    target index is fixed while ``o`` varies). Reports proper set scores
    only: coverage and mean width per ``m``, plus trivial-set shares.
    """
    if int(n_reps) < 1:
        raise ValueError("n_reps must be >= 1")
    ms = sorted({int(m) for m in m_values})
    if not ms or ms[0] < 0:
        raise ValueError("m_values must contain non-negative integers")
    if int(n_groups) < 1:
        raise ValueError("n_groups must be >= 1")
    sizes = int(base_size) + np.arange(int(n_groups), dtype=np.int64)
    target_index = ms[-1]
    if int(sizes.min()) <= target_index:
        raise ValueError("base_size must exceed every m in m_values")

    rng = np.random.default_rng(seed)
    covered: dict[int, list[float]] = {m: [] for m in ms}
    widths: dict[int, list[float]] = {m: [] for m in ms}
    trivials: dict[int, list[float]] = {m: [] for m in ms}
    for rep in range(int(n_reps)):
        effects = rng.normal(0.0, gamma, size=int(n_groups) + 1)
        y_groups = [
            np.asarray(rng.normal(effects[j], sigma, size=int(sizes[j])), dtype=np.float64)
            for j in range(int(n_groups))
        ]
        y_test = np.asarray(rng.normal(effects[-1], sigma, size=target_index + 1), dtype=np.float64)
        y_target = float(y_test[target_index])
        for m in ms:
            res = hcp_predict(y_groups, y_test[:m], alpha=alpha, eta=eta, seed=rep)
            hit = 1.0 if res.lower <= y_target <= res.upper else 0.0
            covered[m].append(hit)
            widths[m].append(float(res.upper - res.lower))
            trivials[m].append(1.0 if res.trivial else 0.0)

    out: dict[str, float | str] = {}
    coverage_means: dict[int, float] = {}
    width_means: dict[int, float] = {}
    for m in ms:
        cov = np.asarray(covered[m], dtype=float)
        wid = np.asarray(widths[m], dtype=float)
        fin = wid[np.isfinite(wid)]
        coverage_means[m] = float(cov.mean())
        width_means[m] = float(fin.mean()) if fin.size > 0 else float("inf")
        out[f"synthetic_coverage_m{m}"] = coverage_means[m]
        out[f"synthetic_mean_width_m{m}"] = width_means[m]
        out[f"synthetic_trivial_share_m{m}"] = float(np.mean(np.asarray(trivials[m], dtype=float)))
    out["synthetic_min_coverage"] = float(min(coverage_means.values()))
    ordered_widths = [width_means[m] for m in ms]
    out["synthetic_width_shrinks"] = 1.0 if ordered_widths[-1] < ordered_widths[0] else 0.0
    out["synthetic_n_reps"] = float(n_reps)
    out["synthetic_alpha"] = float(alpha)
    out["synthetic_eta"] = float(eta)
    out["synthetic_gamma"] = float(gamma)
    out["synthetic_sigma"] = float(sigma)
    out["synthetic_seed"] = float(seed)
    out["synthetic_dgp"] = "synthetic_hierarchical_gaussian"
    out["synthetic_claim"] = "research_metric_only"
    return out
