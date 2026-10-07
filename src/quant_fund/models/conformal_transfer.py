"""Transported Conformal Calibration (TCC) — calibration transfer across spaces.

Implements Doula, "Conformal Calibration Transfer", ICML 2026,
arXiv:2609.10737 (verified against the paper; method summary below follows
its Sections 3.2–3.4 and Appendix B.5–B.6). Setting: labeled calibration
exists only in a source input space ``X_s``; prediction sets with
target-domain coverage are needed in a target space ``X_t`` linked to the
source through unlabeled PAIRED observations ``(x_s, x_t)`` (e.g. paired
modalities, sensors, or venues). Standard split conformal needs labeled
target calibration (unavailable), and Tibshirani et al. (2019) weighted
conformal presumes a common input space (violated when ``X_s != X_t``).

Pipeline (paper Section 3):

1. :func:`transport_calibration` — learn a map ``f: X_s -> X_t`` from the
   unlabeled pairs (paper: encoder–decoder / pix2pix trained with an L1
   reconstruction objective; here the nearest-neighbor-over-pair-features
   instantiation the lane brief permits), form pseudo-target calibration
   ``{(f(x_s), y)}``, and RECOMPUTE target-space scores ``S_t(f(x_s), y)``
   (paper Eqs. 3–4). The paper transports inputs and rescores with the fixed
   target predictor; it does not move score values directly.
2. :func:`tcc_ks` — TCC-KS (paper Section 3.3): one-sided Kolmogorov–Smirnov
   gap ``delta_hat = sup_u {F~_T(u) - F_T(u)}_+`` between a label-free
   uncertainty surrogate ``T`` on transported vs real unlabeled target inputs
   (Eq. 9), inflated by the two-sample DKW/Massart bound ``delta_plus``
   (Eq. 10), giving the adjusted level ``alpha* = max(0, alpha - delta_plus)``
   (Eq. 11). ``alpha* = 0`` uses the paper's footnote-1 convention: the
   threshold is the sample max ``S_(n)``, never ``+inf``. Theorem 3.2 (under
   A1 task invariance, A2 approximate surrogate control, A3 sample
   splitting): target coverage ``>= 1 - (alpha* + delta_plus + eps)``, i.e.
   ``>= 1 - (alpha + eps)`` whenever ``delta_plus <= alpha``.
3. :func:`weighted_tcc` — weighted-TCC (paper Section 3.4, Prop. 3.3):
   estimate the residual density ratio ``w(x) = p_Xt(x) / p_X~t(x)`` with a
   logistic domain classifier (paper App. B.6: odds ``p/(1-p)``, clipped at
   5) and apply the importance-weighted split-conformal quantile of
   :mod:`quant_fund.models.weighted_conformal` to the transported scores.
   Under post-transport covariate shift with oracle weights this attains
   target-domain coverage ``>= 1 - alpha``.

Label-free diagnostics (paper Section 4): ``delta_plus`` (KS certificate) and
``ESS%`` (weight stability) are returned alongside every threshold and
indicate which correction is trustworthy at deployment.

Honesty: everything numeric here is validated on a SYNTHETIC fixture
(:func:`synthetic_paired_shift`, planted Gaussian shift in pair-feature
space) — a correctness test, never market evidence. Research outputs are
proper-score quantities only (coverage, interval width, KS certificate,
ESS); no P&L / Sharpe-family metrics. Determinism: the only randomness lives
inside seeded synthetic generators; all estimators are closed-form.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import log, sqrt

import numpy as np
from numpy.typing import NDArray
from scipy.special import expit

from quant_fund.metrics.conformal import conformal_quantile, expand_interval, set_metrics
from quant_fund.models.weighted_conformal import weighted_conformal_quantile

Array = NDArray[np.float64]

# Weight clip for weighted-TCC odds (paper App. B.6 clips at 5); floor mirrors
# WEIGHT_CLIP[0] in quant_fund.models.weighted_conformal.
WEIGHT_FLOOR: float = 1e-3
DEFAULT_WEIGHT_CLIP: float = 5.0
_CHUNK_ROWS: int = 4096


def _finite_1d(x: object, name: str) -> Array:
    """Fail-closed coercion to a finite, non-empty 1-d float array."""
    arr = np.asarray(x, dtype=float).ravel()
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")
    return arr


def _finite_2d(x: object, name: str) -> Array:
    """Fail-closed coercion to a finite, non-empty 2-d float array.

    A 1-d input is interpreted as ``(n, 1)`` feature rows.
    """
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2 or arr.shape[0] == 0:
        raise ValueError(f"{name} must be a non-empty 2-d array")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")
    return arr


def _pairwise_sqdist(a: Array, b: Array) -> Array:
    """Squared Euclidean distances between rows of ``a`` and rows of ``b``."""
    d2 = np.sum(a * a, axis=1)[:, None] + np.sum(b * b, axis=1)[None, :] - 2.0 * (a @ b.T)
    return np.asarray(np.maximum(d2, 0.0), dtype=float)


class NearestNeighborTransport:
    """Deterministic paired-bridge transport map ``f: X_s -> X_t``.

    Nearest-neighbor-over-pair-features regression: ``f(x)`` is the mean
    target view of the ``k`` closest source views among the unlabeled pairs.
    This is the k-NN counterpart of the paper's learned map (they train an
    encoder–decoder with an L1 objective on images; the lane brief permits
    the nearest-neighbor/kernel instantiation over pair features). k-NN
    averaging is used instead of Nadaraya–Watson kernel regression because
    NW shrinks the identity component of a translation map toward the data
    mean under a Gaussian source marginal, which would fabricate a
    surrogate mismatch even when the domains agree.

    Deterministic: no RNG; ties broken by array order (measure-zero under
    continuous features). Chunked over queries to bound memory.
    """

    def __init__(self, x_src: Array, x_tgt: Array, k: int) -> None:
        self.x_src = x_src
        self.x_tgt = x_tgt
        self.k = int(k)

    @classmethod
    def fit(
        cls, x_src_pairs: Array, x_tgt_pairs: Array, k_neighbors: int | None = None
    ) -> NearestNeighborTransport:
        xs = _finite_2d(x_src_pairs, "x_src_pairs")
        xt = _finite_2d(x_tgt_pairs, "x_tgt_pairs")
        if xs.shape != xt.shape:
            raise ValueError("x_src_pairs and x_tgt_pairs must have the same shape")
        m = xs.shape[0]
        if k_neighbors is None:
            # Cube-root rule: classical k-NN scaling in low dimension. Small k
            # keeps the map faithful (large k over-smooths and shrinks the
            # transported distribution, which would fabricate a surrogate
            # mismatch delta_hat even when the domains agree).
            k = max(3, int(round(float(m) ** (1.0 / 3.0))))
        else:
            k = int(k_neighbors)
        if k < 1:
            raise ValueError("k_neighbors must be >= 1")
        if k > m:
            raise ValueError("k_neighbors cannot exceed the number of pairs")
        return cls(x_src=xs, x_tgt=xt, k=k)

    def predict(self, x: Array) -> Array:
        xq = _finite_2d(x, "x")
        if xq.shape[1] != self.x_src.shape[1]:
            raise ValueError("query features must match the pair feature dimension")
        k = self.k
        out = np.empty((xq.shape[0], self.x_tgt.shape[1]), dtype=float)
        for start in range(0, xq.shape[0], _CHUNK_ROWS):
            block = xq[start : start + _CHUNK_ROWS]
            d2 = _pairwise_sqdist(block, self.x_src)
            idx = np.argpartition(d2, k - 1, axis=1)[:, :k]
            out[start : start + block.shape[0]] = self.x_tgt[idx].mean(axis=1)
        return out


@dataclass(frozen=True)
class TransportedCalibration:
    """Pseudo-target calibration ``{(f(x_s), y)}`` with recomputed scores.

    ``scores[i] = S_t(x_transported[i], y_cal[i])`` (paper Eq. 3–4). ``map``
    is retained so the same ``f`` can be applied to held-out unlabeled pairs
    for the TCC-KS surrogate and weighted-TCC ratio estimation (A3).
    """

    x_transported: Array
    scores: Array
    map: NearestNeighborTransport


def transport_calibration(
    x_src_cal: Array,
    y_cal: Array,
    x_src_pairs: Array,
    x_tgt_pairs: Array,
    score_fn: Callable[[Array, Array], Array],
    k_neighbors: int | None = None,
) -> TransportedCalibration:
    """Transport labeled source calibration into the target space (paper §3.2).

    Learns ``f`` from the unlabeled pairs, maps the source calibration inputs
    to ``x~_t = f(x_s)``, keeps the labels, and recomputes target-space
    nonconformity scores via ``score_fn(x_tgt, y)`` (the fixed target
    predictor plus score of the paper's ``g_t``, ``S_t``). The brief's
    "(score, pair features)" calibration enters as ``(x_src_cal, y_cal)``
    because the paper's transport rescores with the target model rather than
    moving source score values across spaces.

    Fail-closed: raises ``ValueError`` on empty/non-finite inputs, shape
    mismatches, or a ``score_fn`` that does not return one finite score per
    calibration row.
    """
    xs = _finite_2d(x_src_cal, "x_src_cal")
    y = _finite_1d(y_cal, "y_cal")
    if xs.shape[0] != y.size:
        raise ValueError("x_src_cal and y_cal must have the same number of rows")
    tmap = NearestNeighborTransport.fit(x_src_pairs, x_tgt_pairs, k_neighbors=k_neighbors)
    if xs.shape[1] != tmap.x_src.shape[1]:
        raise ValueError("x_src_cal features must match the source pair-feature dimension")
    x_transported = tmap.predict(xs)
    scores = np.asarray(score_fn(x_transported, y), dtype=float).ravel()
    if scores.shape[0] != y.size:
        raise ValueError("score_fn must return one score per calibration row")
    if not np.all(np.isfinite(scores)):
        raise ValueError("score_fn returned non-finite transported scores")
    return TransportedCalibration(x_transported=x_transported, scores=scores, map=tmap)


def one_sided_ks_gap(surrogate_transported: Array, surrogate_target: Array) -> float:
    """One-sided KS discrepancy ``delta_hat`` (paper Eq. 9).

    ``sup_u {F~_T(u) - F_T(u)}_+`` between the empirical CDF of the label-free
    surrogate on transported inputs and on real unlabeled target inputs. The
    positive part keeps only the "target is harder" direction; the opposite
    deviation means the transported calibration is conservative and needs no
    correction. The sup of right-continuous step CDFs is attained at an
    observed value, so the grid is the union of both samples.
    """
    a = _finite_1d(surrogate_transported, "surrogate_transported")
    b = _finite_1d(surrogate_target, "surrogate_target")
    a_sorted = np.sort(a)
    b_sorted = np.sort(b)
    grid = np.union1d(a_sorted, b_sorted)
    cdf_a = np.searchsorted(a_sorted, grid, side="right") / float(a.size)
    cdf_b = np.searchsorted(b_sorted, grid, side="right") / float(b.size)
    return float(max(float(np.max(cdf_a - cdf_b)), 0.0))


@dataclass(frozen=True)
class TCCKSResult:
    """TCC-KS output: threshold plus label-free certificate (paper §3.3)."""

    qhat: float
    alpha_star: float
    delta_hat: float
    delta_plus: float


def tcc_ks(
    transported_scores: Array,
    surrogate_transported: Array,
    surrogate_target: Array,
    alpha: float,
    eta: float = 0.1,
) -> TCCKSResult:
    """TCC-KS: KS-certified conservative level adjustment (paper §3.3, Thm 3.2).

    ``delta_plus = delta_hat + sqrt(log(4/eta)/(2 m_tgt))
    + sqrt(log(4/eta)/(2 m_trans))`` is the two-sample DKW/Massart inflation
    of the one-sided surrogate gap (Eq. 10, ``eta = 0.1`` in the paper's
    experiments). The adjusted internal level is
    ``alpha* = max(0, alpha - delta_plus)`` (Eq. 11) and the threshold is the
    split-conformal order statistic of the transported labeled scores at
    ``alpha*`` (Vovk convention via
    :func:`quant_fund.metrics.conformal.conformal_quantile`). When
    ``alpha* = 0`` the paper's footnote-1 convention applies: use the sample
    max ``S_(n)`` instead of the vacuous ``+inf`` threshold.

    Coverage (Thm 3.2, under A1/A2/A3): with probability ``>= 1 - eta`` over
    the unlabeled samples, target coverage ``>= 1 - (alpha* + delta_plus +
    eps) >= 1 - (alpha + eps)`` whenever ``delta_plus <= alpha``.

    Fail-closed: raises ``ValueError`` on empty/non-finite inputs or
    ``alpha``/``eta`` outside ``(0, 1)``.
    """
    s = _finite_1d(transported_scores, "transported_scores")
    a = _finite_1d(surrogate_transported, "surrogate_transported")
    b = _finite_1d(surrogate_target, "surrogate_target")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if not 0.0 < eta < 1.0:
        raise ValueError("eta must be in (0, 1)")
    delta_hat = one_sided_ks_gap(a, b)
    inflation = sqrt(log(4.0 / eta) / (2.0 * float(b.size))) + sqrt(
        log(4.0 / eta) / (2.0 * float(a.size))
    )
    delta_plus = delta_hat + inflation
    alpha_star = max(0.0, alpha - delta_plus)
    if alpha_star == 0.0:
        qhat = float(np.max(s))
    else:
        qhat = conformal_quantile(s, alpha_star)
    return TCCKSResult(qhat=qhat, alpha_star=alpha_star, delta_hat=delta_hat, delta_plus=delta_plus)


def _irls_logistic(
    x: Array, labels: Array, l2: float, max_iter: int = 100, tol: float = 1e-10
) -> Array:
    """Ridge-penalized logistic regression by IRLS/Newton from a zero start.

    Deterministic (no RNG); the small ridge keeps the Hessian nonsingular
    under near-separation, and divergent iterates fail closed. Returns the
    coefficient vector ``[intercept, *slopes]``.
    """
    n, d = x.shape
    design = np.hstack([np.ones((n, 1)), x])
    reg = np.full(d + 1, float(l2))
    reg[0] = 0.0
    beta = np.zeros(d + 1)
    for _ in range(int(max_iter)):
        p = np.asarray(expit(design @ beta), dtype=float)
        curvature = np.maximum(p * (1.0 - p), 1e-9)
        grad = design.T @ (labels - p) - reg * beta
        hess = (design * curvature[:, None]).T @ design + np.diag(reg)
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError as exc:
            raise ValueError("domain classifier Hessian is singular") from exc
        beta = beta + step
        if not np.all(np.isfinite(beta)):
            raise ValueError("domain classifier diverged")
        if float(np.max(np.abs(step))) < tol:
            break
    return beta


def domain_ratio_weights(
    feats_rows: Array,
    feats_transported_pool: Array,
    feats_target_pool: Array,
    weight_clip: float = DEFAULT_WEIGHT_CLIP,
) -> Array:
    """Residual density-ratio weights ``w(x) ~ p_Xt(x) / p_X~t(x)`` (paper §3.4).

    Logistic domain classifier (label 1 = real target pool, 0 = transported
    pool) on target-space features; per paper App. B.6 the weight is the odds
    ``p/(1-p)`` — corrected here by the class-prior factor ``n_trans/n_tgt``
    so the odds estimate the ratio even for unequal pool sizes — clipped to
    ``[WEIGHT_FLOOR, weight_clip]`` (the paper clips at 5). Features are
    standardized jointly before the fit for numerical stability.

    ``feats_rows`` are the transported calibration rows the weights will be
    applied to; the two pools train the classifier on unlabeled data only.
    Fail-closed on shape/feature-dimension mismatches, empty or non-finite
    inputs, and ``weight_clip <= WEIGHT_FLOOR``.
    """
    rows = _finite_2d(feats_rows, "feats_rows")
    x0 = _finite_2d(feats_transported_pool, "feats_transported_pool")
    x1 = _finite_2d(feats_target_pool, "feats_target_pool")
    if not np.isfinite(weight_clip) or float(weight_clip) <= WEIGHT_FLOOR:
        raise ValueError(f"weight_clip must be finite and > {WEIGHT_FLOOR}")
    if x0.shape[1] != x1.shape[1] or rows.shape[1] != x0.shape[1]:
        raise ValueError("feature blocks must share the same feature dimension")
    pool = np.vstack([x0, x1])
    labels = np.concatenate([np.zeros(x0.shape[0]), np.ones(x1.shape[0])])
    mu = pool.mean(axis=0)
    sd = pool.std(axis=0)
    sd = np.where(sd > 0.0, sd, 1.0)
    beta = _irls_logistic((pool - mu) / sd, labels, l2=1e-2)
    z = (rows - mu) / sd
    logit = np.asarray(z @ beta[1:] + beta[0], dtype=float)
    p = np.clip(np.asarray(expit(logit), dtype=float), 1e-12, 1.0 - 1e-12)
    odds = (p / (1.0 - p)) * (float(x0.shape[0]) / float(x1.shape[0]))
    return np.asarray(np.clip(odds, WEIGHT_FLOOR, float(weight_clip)), dtype=float)


@dataclass(frozen=True)
class WeightedTCCResult:
    """Weighted-TCC output: threshold, weights, and the ESS% diagnostic."""

    qhat: float
    weights: Array
    ess_percent: float


def weighted_tcc(
    transported_scores: Array,
    feats_scores: Array,
    feats_transported_pool: Array,
    feats_target_pool: Array,
    alpha: float,
    weight_clip: float = DEFAULT_WEIGHT_CLIP,
) -> WeightedTCCResult:
    """Weighted-TCC: transport-then-reweight correction (paper §3.4, Prop. 3.3).

    Reweights the transported labeled calibration scores toward the real
    target input distribution using
    :func:`domain_ratio_weights`, then takes the importance-weighted
    split-conformal quantile of
    :func:`quant_fund.models.weighted_conformal.weighted_conformal_quantile`
    (Tibshirani et al. 2019 order-statistic convention). Under post-transport
    covariate shift with oracle weights, target coverage ``>= 1 - alpha``
    (Prop. 3.3); with estimated weights, reliability tracks the label-free
    ``ess_percent = 100 * (sum w)^2 / (n * sum w^2)`` diagnostic (paper §4).

    Fail-closed: raises ``ValueError`` on empty/non-finite inputs, one row of
    ``feats_scores`` per transported score not satisfied, ``alpha`` outside
    ``(0, 1)``, or an invalid ``weight_clip``.
    """
    s = _finite_1d(transported_scores, "transported_scores")
    rows = _finite_2d(feats_scores, "feats_scores")
    if rows.shape[0] != s.size:
        raise ValueError("feats_scores must have one row per transported score")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    w = domain_ratio_weights(rows, feats_transported_pool, feats_target_pool, weight_clip)
    qhat = weighted_conformal_quantile(s, w, alpha)
    ess = float(w.sum()) ** 2 / (float(w.size) * float(np.sum(w * w)))
    return WeightedTCCResult(qhat=qhat, weights=w, ess_percent=100.0 * ess)


@dataclass(frozen=True)
class LinearTargetModel:
    """Fixed target predictor for the SYNTHETIC fixture (paper's ``g_t``).

    The paired views are related by a coordinate shift ``x_t = z + shift``
    (same latent instance ``z``, same label). With ``u = x - shift``:

    - ``predict(x)   = coef . u`` — exact in target space;
    - ``surrogate(x) = 1 + kappa * ||u||`` — the label-free difficulty proxy
      ``T`` (regression analogue of the paper's least-confidence
      ``1 - max_y p_t(y|x)``: computable from the input alone, no labels);
    - ``score(x, y)  = |y - predict(x)|`` — the nonconformity ``S_t``.

    ``features`` returns ``[predict, T, T^2]`` for the domain classifier:
    the quadratic surrogate expansion lets the logistic odds represent the
    exact Gaussian density-ratio log-odds of the fixture (the paper uses
    target-predictor logits as classifier features, App. B.6).
    """

    coef: Array
    shift: float
    kappa: float

    def _check(self, x: Array) -> Array:
        x2 = _finite_2d(x, "x")
        if x2.shape[1] != self.coef.size:
            raise ValueError("x must have one column per model coefficient")
        return x2

    def predict(self, x: Array) -> Array:
        return np.asarray((self._check(x) - self.shift) @ self.coef, dtype=float)

    def surrogate(self, x: Array) -> Array:
        u = self._check(x) - self.shift
        return np.asarray(1.0 + self.kappa * np.linalg.norm(u, axis=1), dtype=float)

    def score(self, x: Array, y: Array) -> Array:
        yy = _finite_1d(y, "y")
        pred = self.predict(x)
        if pred.size != yy.size:
            raise ValueError("y must have one value per input row")
        return np.asarray(np.abs(yy - pred), dtype=float)

    def features(self, x: Array) -> Array:
        t = self.surrogate(x)
        return np.asarray(np.column_stack([self.predict(x), t, t * t]), dtype=float)


@dataclass(frozen=True)
class SyntheticTCCData:
    """SYNTHETIC paired-domain fixture. Correctness test, not market evidence.

    Sample usage mirrors paper Assumption A3: ``*_pairs_fit`` learns the
    transport map; the disjoint ``*_pairs_eval`` provides the unlabeled
    samples for the KS surrogate and the density-ratio classifier;
    ``x_tgt_test``/``y_test`` are the labeled target test set used for
    EVALUATION ONLY.
    """

    model: LinearTargetModel
    x_src_cal: Array
    y_cal: Array
    x_src_pairs_fit: Array
    x_tgt_pairs_fit: Array
    x_src_pairs_eval: Array
    x_tgt_pairs_eval: Array
    x_tgt_test: Array
    y_test: Array


def synthetic_paired_shift(
    n_cal: int = 1500,
    n_pairs_fit: int = 1500,
    n_pairs_eval: int = 2000,
    n_test: int = 4000,
    *,
    shift: float = 2.0,
    kappa: float = 1.5,
    target_scale: float = 1.6,
    dim: int = 2,
    seed: int = 0,
) -> SyntheticTCCData:
    """Seeded SYNTHETIC source/target fixture with a controlled mismatch knob.

    Latent instances ``z``; paired views ``x_s = z`` and ``x_t = z + shift``
    share the label ``y = coef.z + (1 + kappa*||z||) * noise``. Calibration
    and transport-fit pair latents are ``N(0, I_dim)``; deployment target
    latents (the unlabeled eval target pool AND the labeled test set) are
    ``N(0, target_scale^2 * I_dim)`` — a planted Gaussian shift in
    pair-feature space, observable by both corrections through the unlabeled
    deployment pool (no target labels). Target-space scores are
    ``S = (1 + kappa*||z||) *
    |noise|`` and the surrogate ``T(x) = 1 + kappa*||x - shift||`` tracks
    them (paper A2 with small slack), so with ``target_scale > 1`` the
    transported calibration UNDERCOVERS — the gap TCC-KS and weighted-TCC
    correct — while ``target_scale = 1`` is the no-mismatch regime where both
    must degrade gracefully to standard (weighted) split conformal.
    """
    sizes = {
        "n_cal": n_cal,
        "n_pairs_fit": n_pairs_fit,
        "n_pairs_eval": n_pairs_eval,
        "n_test": n_test,
    }
    for name, value in sizes.items():
        if int(value) < 1:
            raise ValueError(f"{name} must be >= 1")
    if not np.isfinite(shift):
        raise ValueError("shift must be finite")
    if not np.isfinite(kappa) or float(kappa) < 0.0:
        raise ValueError("kappa must be finite and >= 0")
    if not np.isfinite(target_scale) or float(target_scale) <= 0.0:
        raise ValueError("target_scale must be finite and > 0")
    if int(dim) < 1:
        raise ValueError("dim must be >= 1")

    rng = np.random.default_rng(seed)
    coef = np.full(int(dim), 1.0 / sqrt(float(dim)))
    model = LinearTargetModel(coef=coef, shift=float(shift), kappa=float(kappa))

    def draw_z(n: int, scale: float) -> Array:
        return np.asarray(rng.normal(size=(int(n), int(dim))) * float(scale), dtype=float)

    def labels(z: Array) -> Array:
        difficulty = 1.0 + float(kappa) * np.linalg.norm(z, axis=1)
        noise = rng.normal(size=z.shape[0])
        return np.asarray(z @ coef + difficulty * noise, dtype=float)

    z_cal = draw_z(n_cal, 1.0)
    z_fit = draw_z(n_pairs_fit, 1.0)
    # Eval samples (A3: disjoint from the fit pairs, unlabeled). The KS test
    # and the ratio classifier use MARGINALS only (paper Eqs. 7-9, App. B.6):
    # the source side matches the calibration population (pushed through f-hat
    # to form transported surrogate/weight samples) and the target side is the
    # deployment target pool, so cross pairing is immaterial here.
    z_eval_src = draw_z(n_pairs_eval, 1.0)
    z_eval_tgt = draw_z(n_pairs_eval, target_scale)
    z_test = draw_z(n_test, target_scale)
    return SyntheticTCCData(
        model=model,
        x_src_cal=z_cal,
        y_cal=labels(z_cal),
        x_src_pairs_fit=z_fit,
        x_tgt_pairs_fit=z_fit + shift,
        x_src_pairs_eval=z_eval_src,
        x_tgt_pairs_eval=z_eval_tgt + shift,
        x_tgt_test=z_test + shift,
        y_test=labels(z_test),
    )


def bench_tcc(
    n_cal: int = 1500,
    n_pairs_fit: int = 2000,
    n_pairs_eval: int = 4000,
    n_test: int = 4000,
    alpha: float = 0.10,
    eta: float = 0.10,
    target_scale: float = 1.4,
    weight_clip: float = DEFAULT_WEIGHT_CLIP,
    seed: int = 11,
) -> dict[str, float | str]:
    """Coverage and width of the TCC variants on the planted Gaussian pair shift.

    SYNTHETIC correctness fixture (proper scores only: coverage, interval
    width, KS certificate, ESS). No P&L / Sharpe-family keys; not market
    evidence. Composes the full pipeline: transport (§3.2) -> TCC-KS (§3.3)
    and weighted-TCC (§3.4) on disjoint unlabeled eval pairs (A3), evaluated
    on the labeled target test set via
    :mod:`quant_fund.metrics.conformal` interval helpers.
    """
    data = synthetic_paired_shift(
        n_cal,
        n_pairs_fit,
        n_pairs_eval,
        n_test,
        target_scale=target_scale,
        seed=seed,
    )
    model = data.model
    transported = transport_calibration(
        data.x_src_cal,
        data.y_cal,
        data.x_src_pairs_fit,
        data.x_tgt_pairs_fit,
        model.score,
    )
    x_eval_transported = transported.map.predict(data.x_src_pairs_eval)
    ks = tcc_ks(
        transported.scores,
        model.surrogate(x_eval_transported),
        model.surrogate(data.x_tgt_pairs_eval),
        alpha,
        eta=eta,
    )
    wtcc = weighted_tcc(
        transported.scores,
        model.features(transported.x_transported),
        model.features(x_eval_transported),
        model.features(data.x_tgt_pairs_eval),
        alpha,
        weight_clip=weight_clip,
    )
    q_plain = conformal_quantile(transported.scores, alpha)
    point = model.predict(data.x_tgt_test)
    out: dict[str, float | str] = {}
    for tag, q in (("transport_only", q_plain), ("tcc_ks", ks.qhat), ("weighted_tcc", wtcc.qhat)):
        lo, hi = expand_interval(point, point, q)
        row = set_metrics(data.y_test, lo, hi)
        out[f"synthetic_coverage_{tag}"] = row.coverage
        out[f"synthetic_mean_width_{tag}"] = row.mean_width
        out[f"synthetic_qhat_{tag}"] = float(q)
    out.update(
        {
            "synthetic_delta_hat": ks.delta_hat,
            "synthetic_delta_plus": ks.delta_plus,
            "synthetic_alpha_star": ks.alpha_star,
            "synthetic_ess_percent": wtcc.ess_percent,
            "synthetic_n": float(data.y_test.size),
            "synthetic_alpha": float(alpha),
            "synthetic_target_scale": float(target_scale),
            "synthetic_dgp": "fixture",
            "synthetic_claim": "research_metric_only",
            "synthetic_seed": float(seed),
        }
    )
    return out
