"""MS-RLCP: multi-source randomly localized conformal prediction. No Sharpe.

Implements Multi-Source Randomly Localized Conformal Prediction (MS-RLCP) of
Hore, Chatterjee & Choudhury (2026), "Multi-source conformal prediction:
leveraging heterogeneity via localization", arXiv:2609.14531 [stat.ML] (the
"MS paper"), built on randomly localized conformal prediction (RLCP) of Hore &
Barber (2025), "Conformal prediction with local weights: randomization enables
robust guarantees", JRSS-B 87(2), 549-578, arXiv:2310.07850 ("HB25"), which
extends Guan (2023, Biometrika 110(1), 33-50) localized CP and Tibshirani,
Barber, Candes & Ramdas (2019, NeurIPS 32) weighted CP.

Setting (MS paper Sec. 1.1): K heterogeneous sources D_k = {(Y_i,k, X_i,k)},
independent across k; the test distribution P_test may differ from every
source; under covariate shift P_k = P_{Y|X} x P_{k,X} and P_test = P_{Y|X} x
P_test,X the conditional response law P_{Y|X} is shared while feature
marginals differ.

RLCP (MS paper Sec. 2.2, HB25): a symmetric localization kernel H(x, .), a
density in its second argument, perturbs the test feature,
X~_{n+1} ~ H(X_{n+1}, .), and the calibration scores are weighted by kernel
values centered at the *perturbed* feature (eq. 2.5):

    w_i = H(X_i, X~_{n+1}) / Z,
    Z = H(X_{n+1}, X~_{n+1}) + sum_i H(X_i, X~_{n+1}),

with qhat the (1 - alpha)-quantile of sum_i w_i delta_{S_i} + w_{n+1}
delta_{+inf}, w_{n+1} = H(X_{n+1}, X~_{n+1}) / Z. This gives the relaxed
local guarantee P(Y in C | X~, D_train) >= 1 - alpha (MS paper Lemma 2.1 =
HB25 Prop. 1): coverage conditional on a random neighborhood of the test
point. The kernels of MS paper eq. (2.4) on X = R^d, bandwidth h > 0:

    Gaussian: H(x, y) = (2 pi)^{-d/2} h^{-d} exp(-||x - y||_2^2 / (2 h^2)),
              X~ = X + h Z, Z ~ N_d(0, I_d);
    box:      H(x, y) = 1{||x - y||_2 <= h} / (V_d h^d), X~ uniform on the
              closed Euclidean ball of radius h at X, V_d = pi^{d/2} /
              Gamma(d/2 + 1) the unit-ball volume.

MS-RLCP (MS paper Alg. 1): (I) each source k keeps its own train/cal split
and its own score s_k fitted on train (no data sharing, Remark 2.1); (II)
data-adaptive source selection -- the alignment score
what_k = (1 / n_{k,train}) sum_{D_k,train} H(X, X~_{n+1}) (line 4) estimates
the perturbed-feature density E_{X ~ P_k,X}[H(X, X~)] (eq. 2.9) and
k_hat = argmax_k what_k (line 6, ties by the fixed lowest-index rule); (III)
the RLCP weighted quantile of the selected source's calibration scores
(lines 7-10) defines C(X_{n+1}) = {y : s_{k_hat}(X_{n+1}, y) <= qhat}
(line 11).

Envelope guarantee (MS paper Sec. 3): with source feature densities f_k w.r.t.
a common dominating measure nu, the envelope density is
f_bar(x) = B^{-1} max_k f_k(x), B = int max_k f_k d nu (Def. 1); each P_k,X is
absolutely continuous w.r.t. the envelope with max_k ||g_k||_inf = B,
g_k = dP_k,X / dP_bar (Lemma 3.1). If P_test,X << P_bar (Assumption 1 --
strictly weaker than being a mixture of the sources, Lemma 3.2) and the
ratios g_1..g_K, g_test are globally L-Lipschitz with ||g_test||_inf < inf,
Theorem 3.3 gives, with n_eff = min_k n_{k,train} >= 2,

    P(Y in C) >= (1 - alpha - 1 / n_eff)
        - 2 L (1 + 2 ||g_test||_inf / B) E_{P_bar} ||X - X~||_2
        - P_{P_test, X~ | X ~ H(X,.)}(max_k E_{P_k,X}[H(X', X~)] <= t_n_eff),

    t_n_eff = 2 ||H||_inf sqrt(2 log(2 K n_eff) / n_eff).

The second term is O(h) for the box kernel (Cor. 3.4); under additional
regularity the test-conditional coverage converges to 1 - alpha (Thm. 3.5).

Honesty: ``envelope_coverage_bound`` is a PLUG-IN DIAGNOSTIC of Theorem 3.3,
not a proof certificate: the representation term replaces the population
expectations mu_k(X~) by the same train-sample averages Algorithm 1 line 4
uses and the probability by a seeded Monte Carlo mean over the supplied test
features, while B, ||g_test||_inf and L are oracle constants --
``envelope_constants_1d`` estimates them by grid integration of supplied true
densities, so it is valid only for SYNTHETIC correctness checks, never market
evidence. A bound <= 0 is flagged ``vacuous`` (degradation surfaced, never
hidden), and an unattainable level yields qhat = +inf reported as a vacuous
prediction set, never clipped to a finite width. Coverage, set width, flag
rates and bound terms are proper calibration diagnostics; no Sharpe-family
headline is produced here. The smoothed RLCP variant of HB25 App. B (used in
the MS paper's numerical experiments to reduce over-coverage) is deliberately
not implemented; this module follows plain Alg. 1.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.special import gamma as _gamma

Array = NDArray[np.float64]
DensityFn = Callable[[Array], Array]
RNG = np.random.Generator | int

#: Kernel kinds of MS paper eq. (2.4).
KERNEL_KINDS: tuple[str, str] = ("gaussian", "box")


def _validate_alpha(alpha: float) -> float:
    a = float(alpha)
    if not np.isfinite(a) or not 0.0 < a < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    return a


def _finite_1d(values: object, name: str) -> Array:
    v = np.asarray(values, dtype=float).reshape(-1)
    if v.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if not bool(np.all(np.isfinite(v))):
        raise ValueError(f"{name} must be finite")
    return v


def _as_2d(values: object, name: str) -> Array:
    a = np.asarray(values, dtype=float)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    if a.ndim != 2 or a.shape[0] == 0:
        raise ValueError(f"{name} must be a non-empty (n, d) feature array")
    if not bool(np.all(np.isfinite(a))):
        raise ValueError(f"{name} must be finite")
    return a


def _resolve_rng(rng: RNG | None) -> np.random.Generator:
    if rng is None:
        raise ValueError("pass rng (seed int or Generator) or an explicit x_tilde")
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)


@dataclass(frozen=True)
class LocalizationKernel:
    """Localization kernel H of MS paper eq. (2.4) with its X~ sampler.

    ``kind`` is ``"gaussian"`` (X~ = X + h Z, Z ~ N_d(0, I)) or ``"box"``
    (X~ uniform on the closed ball of radius h at X). ``bandwidth`` h > 0;
    ``dim`` = d >= 1. Fail-closed on any other configuration.
    """

    kind: str
    bandwidth: float
    dim: int = 1

    def __post_init__(self) -> None:
        if self.kind not in KERNEL_KINDS:
            raise ValueError(f"kernel kind must be one of {KERNEL_KINDS}")
        h = float(self.bandwidth)
        if not np.isfinite(h) or h <= 0.0:
            raise ValueError("bandwidth must be finite and > 0")
        object.__setattr__(self, "bandwidth", h)
        d = int(self.dim)
        if d < 1:
            raise ValueError("dim must be >= 1")
        object.__setattr__(self, "dim", d)

    @property
    def h_sup(self) -> float:
        """||H||_inf: the kernel value at x = y (its maximum)."""
        h, d = self.bandwidth, self.dim
        if self.kind == "gaussian":
            return float((2.0 * math.pi) ** (-d / 2.0) * h ** (-float(d)))
        v_d = math.pi ** (d / 2.0) / float(_gamma(d / 2.0 + 1.0))
        return float(1.0 / (v_d * h**d))

    @property
    def mean_perturb_distance(self) -> float:
        """E ||X - X~||_2 under X~ | X ~ H(X, .) (Theorem 3.3 second term).

        Distribution-free in X for both kernels: Gaussian h * E||Z||_2 with
        E||Z||_2 = sqrt(2) Gamma((d+1)/2) / Gamma(d/2), Z ~ N_d(0, I); box
        h * d / (d + 1) (radius h U^{1/d}).
        """
        h, d = self.bandwidth, self.dim
        if self.kind == "gaussian":
            ratio = float(_gamma((d + 1) / 2.0)) / float(_gamma(d / 2.0))
            return float(h * math.sqrt(2.0) * ratio)
        return float(h * d / (d + 1))

    def h_matrix(self, a: Array, b: Array) -> Array:
        """Pairwise kernel values H(a_i, b_j), shape (n_a, n_b)."""
        aa = _as_2d(a, "a")
        bb = _as_2d(b, "b")
        if aa.shape[1] != bb.shape[1]:
            raise ValueError("a and b must have the same feature dimension")
        if aa.shape[1] != self.dim:
            raise ValueError(f"features must have dim {self.dim} matching the kernel")
        diff = aa[:, None, :] - bb[None, :, :]
        d2 = np.sum(diff * diff, axis=-1)
        h = self.bandwidth
        if self.kind == "gaussian":
            out = self.h_sup * np.exp(-d2 / (2.0 * h * h))
        else:
            out = np.where(d2 <= h * h, self.h_sup, 0.0)
        return np.asarray(out, dtype=float)

    def h_rows(self, a: Array, b: Array) -> Array:
        """Rowwise kernel values H(a_i, b_i), shape (n,)."""
        aa = _as_2d(a, "a")
        bb = _as_2d(b, "b")
        if aa.shape != bb.shape:
            raise ValueError("a and b must have the same shape")
        if aa.shape[1] != self.dim:
            raise ValueError(f"features must have dim {self.dim} matching the kernel")
        diff = aa - bb
        d2 = np.sum(diff * diff, axis=-1)
        h = self.bandwidth
        if self.kind == "gaussian":
            out = self.h_sup * np.exp(-d2 / (2.0 * h * h))
        else:
            out = np.where(d2 <= h * h, self.h_sup, 0.0)
        return np.asarray(out, dtype=float)

    def perturb(self, x: Array, rng: np.random.Generator) -> Array:
        """Draw X~ | X ~ H(X, .) rowwise (MS paper eq. 2.7)."""
        xx = _as_2d(x, "x")
        if xx.shape[1] != self.dim:
            raise ValueError(f"features must have dim {self.dim} matching the kernel")
        h, d = self.bandwidth, self.dim
        if self.kind == "gaussian":
            return np.asarray(xx + h * rng.standard_normal(xx.shape), dtype=float)
        z = rng.standard_normal(xx.shape)
        nrm = np.linalg.norm(z, axis=1, keepdims=True)
        zero = (nrm == 0.0).ravel()
        if bool(np.any(zero)):  # probability-zero guard: keep the draw in the ball
            z = np.where(zero[:, None], np.zeros_like(z), z)
            z[zero, 0] = 1.0
            nrm = np.linalg.norm(z, axis=1, keepdims=True)
        r = h * rng.random(xx.shape[0]) ** (1.0 / d)
        return np.asarray(xx + z / nrm * r[:, None], dtype=float)


def localization_kernel(kind: str, bandwidth: float, dim: int = 1) -> LocalizationKernel:
    """Validated constructor for :class:`LocalizationKernel`."""
    return LocalizationKernel(kind=str(kind), bandwidth=float(bandwidth), dim=int(dim))


def alignment_threshold(h_sup: float, n_sources: int, n_eff: int) -> float:
    """t_n_eff = 2 ||H||_inf sqrt(2 log(2 K n_eff) / n_eff) (MS paper Thm. 3.3).

    The concentration threshold of Lemma B.2 (Hoeffding + union bound over the
    K sources at delta = 1 / n_eff). Fail-closed: the theorem requires
    n_eff >= 2 and a bounded kernel.
    """
    hs = float(h_sup)
    if not np.isfinite(hs) or hs <= 0.0:
        raise ValueError("h_sup must be finite and > 0")
    k, ne = int(n_sources), int(n_eff)
    if k < 1:
        raise ValueError("n_sources must be >= 1")
    if ne < 2:
        raise ValueError("n_eff must be >= 2 (MS paper Thm. 3.3)")
    return float(2.0 * hs * math.sqrt(2.0 * math.log(2.0 * k * ne) / ne))


def rlcp_quantile(scores: Array, weights: Array, w_inf: float, alpha: float) -> float:
    """(1 - alpha)-quantile of sum_i w_i delta_{S_i} + w_inf delta_{+inf}.

    Weighted conformal quantile with an explicit +inf atom (MS paper eq. 2.5 /
    Alg. 1 line 10; Tibshirani et al. 2019): qhat = inf{t : F_w(t) >= 1 -
    alpha}. Weights need not be pre-normalized. When the level is unattainable
    from the finite scores (the +inf atom carries more than alpha of the
    mass), returns ``+inf`` -- the vacuous prediction set, reported as-is,
    never clipped. With uniform weights and w_inf = sum(w) / n this coincides
    exactly with the repo's ``localized_conformal_quantile`` (whose implicit
    +inf atom has weight 1 / (n + 1)) and hence with ``conformal_quantile``.
    """
    s = _finite_1d(scores, "scores")
    w = _finite_1d(weights, "weights")
    if s.size != w.size:
        raise ValueError("scores and weights must have the same length")
    if bool(np.any(w < 0.0)):
        raise ValueError("weights must be non-negative")
    wi = float(w_inf)
    if not np.isfinite(wi) or wi < 0.0:
        raise ValueError("w_inf must be finite and >= 0")
    a = _validate_alpha(alpha)
    order = np.argsort(s, kind="mergesort")
    cum = np.cumsum(w[order])
    total = float(cum[-1]) + wi
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("weights and w_inf must carry positive total mass")
    idx = int(np.searchsorted(cum, (1.0 - a) * total, side="left"))
    if idx >= s.size:
        return float("inf")
    return float(s[order][idx])


def _quantile_rows(scores: Array, weights: Array, w_inf: Array, alpha: float) -> Array:
    """Rowwise :func:`rlcp_quantile` on a shared score vector (bit-identical)."""
    a = _validate_alpha(alpha)
    order = np.argsort(scores, kind="mergesort")
    w_sorted = weights[:, order]
    cum = np.cumsum(w_sorted, axis=1)
    total = cum[:, -1] + w_inf
    hit = cum >= (1.0 - a) * total[:, None]
    s_sorted = scores[order]
    out = np.where(hit.any(axis=1), s_sorted[np.argmax(hit, axis=1)], np.inf)
    return np.asarray(out, dtype=float)


def _resolve_x_tilde(
    kernel: LocalizationKernel, x_test: Array, rng: RNG | None, x_tilde: Array | None
) -> Array:
    if x_tilde is not None:
        xt = _as_2d(x_tilde, "x_tilde")
        if xt.shape != x_test.shape:
            raise ValueError("x_tilde must have the same shape as x_test")
        return xt
    return kernel.perturb(x_test, _resolve_rng(rng))


@dataclass(frozen=True)
class RLCPWeightedScores:
    """Weighted empirical score distribution of MS paper eq. (2.5).

    ``weights[j, i]`` = w_i for test point j (calibration rows, normalized by
    Z_j), ``w_inf[j]`` = w_{n+1} (the +inf atom), ``x_tilde[j]`` = X~_j.
    """

    cal_scores: Array
    weights: Array
    w_inf: Array
    x_tilde: Array

    def quantile(self, alpha: float) -> Array:
        """Per-test-point qhat; +inf where the level is unattainable."""
        return _quantile_rows(self.cal_scores, self.weights, self.w_inf, alpha)


def rlcp_scores(
    x_cal: Array,
    cal_scores: Array,
    x_test: Array,
    *,
    kernel: LocalizationKernel,
    rng: RNG | None = None,
    x_tilde: Array | None = None,
) -> RLCPWeightedScores:
    """Single-source RLCP kernel weights (MS paper Sec. 2.2, HB25 Sec. 3).

    Perturbs each test feature X~_j ~ H(X_j, .) (seeded ``rng`` or an explicit
    ``x_tilde``; one is required -- the module never touches an unseeded global
    stream) and forms w_i = H(X_i, X~_j) / Z_j with the +inf atom
    w_{n+1} = H(X_j, X~_j) / Z_j (eq. 2.5). Fail-closed on shape/dim
    mismatches, non-finite inputs, empty calibration, or vanishing Z (possible
    only for user-supplied ``x_tilde`` outside the box-kernel support -- the
    prediction set is undefined there, so we refuse rather than guess).
    """
    xc = _as_2d(x_cal, "x_cal")
    s = _finite_1d(cal_scores, "cal_scores")
    if s.size != xc.shape[0]:
        raise ValueError("cal_scores must have one value per calibration row")
    xt = _as_2d(x_test, "x_test")
    if xt.shape[1] != xc.shape[1]:
        raise ValueError("x_cal and x_test must have the same feature dimension")
    x_til = _resolve_x_tilde(kernel, xt, rng, x_tilde)
    h_cal = kernel.h_matrix(x_til, xc)  # (m, n): H(X~_j, X_i), symmetric kernel
    h_test = kernel.h_rows(xt, x_til)  # (m,): H(X_{n+1}, X~_{n+1})
    z = h_test + h_cal.sum(axis=1)
    if not bool(np.all(np.isfinite(z))) or bool(np.any(z <= 0.0)):
        raise ValueError(
            "kernel weights vanish (Z <= 0): the weighted score distribution is "
            "undefined -- widen the bandwidth or check x_tilde"
        )
    weights = np.asarray(h_cal / z[:, None], dtype=float)
    w_inf = np.asarray(h_test / z, dtype=float)
    return RLCPWeightedScores(cal_scores=s, weights=weights, w_inf=w_inf, x_tilde=x_til)


@dataclass(frozen=True)
class RLCPPrediction:
    """Single-source RLCP prediction output (MS paper eq. 2.6)."""

    qhat: Array
    x_tilde: Array
    w_inf: Array
    covered: Array | None
    n_vacuous: int


def rlcp_predict(
    x_cal: Array,
    cal_scores: Array,
    x_test: Array,
    *,
    alpha: float,
    kernel: LocalizationKernel,
    test_scores: Array | None = None,
    rng: RNG | None = None,
    x_tilde: Array | None = None,
) -> RLCPPrediction:
    """RLCP thresholds qhat(X_j, X~_j) for a batch of test features.

    ``test_scores`` are s(X_j, Y_j) under the (externally fitted) score
    function; when given, ``covered[j]`` = 1{s_j <= qhat_j} (a qhat of +inf
    covers -- the vacuous set is honest, and ``n_vacuous`` counts it).
    """
    ws = rlcp_scores(x_cal, cal_scores, x_test, kernel=kernel, rng=rng, x_tilde=x_tilde)
    q = ws.quantile(alpha)
    covered: Array | None = None
    if test_scores is not None:
        ts = _finite_1d(test_scores, "test_scores")
        if ts.size != q.size:
            raise ValueError("test_scores must have one value per test row")
        covered = np.asarray(ts <= q, dtype=float)
    return RLCPPrediction(
        qhat=q,
        x_tilde=ws.x_tilde,
        w_inf=ws.w_inf,
        covered=covered,
        n_vacuous=int(np.count_nonzero(~np.isfinite(q))),
    )


@dataclass(frozen=True)
class SourceCalibration:
    """One source's Alg. 1 splits: alignment train rows + calibration scores.

    ``x_train``: features of D_k,train, used only for the source alignment of
    line 4 (the empirical estimate of E_{P_k,X}[H(X, X~)] of eq. 2.9).
    ``x_cal`` / ``cal_scores``: features X_i and scores S_i = s_k(X_i, Y_i) of
    D_k,cal, where s_k is fitted on D_k,train by the caller (line 3; the
    FitScore routine stays external -- sources never share data, Remark 2.1).
    Fail-closed: non-empty finite arrays with matching lengths.
    """

    x_train: Array
    x_cal: Array
    cal_scores: Array

    def __post_init__(self) -> None:
        xtr = _as_2d(self.x_train, "x_train")
        xc = _as_2d(self.x_cal, "x_cal")
        s = _finite_1d(self.cal_scores, "cal_scores")
        if s.size != xc.shape[0]:
            raise ValueError("cal_scores must have one value per x_cal row")
        if xtr.shape[1] != xc.shape[1]:
            raise ValueError("x_train and x_cal must have the same feature dimension")
        object.__setattr__(self, "x_train", xtr)
        object.__setattr__(self, "x_cal", xc)
        object.__setattr__(self, "cal_scores", s)


@dataclass(frozen=True)
class MSRLCPPrediction:
    """MS-RLCP output (MS paper Alg. 1) with per-point honesty flags.

    ``selected[j]`` = k_hat (0-based, ties by lowest index -- the fixed rule
    of Sec. 2.3); ``alignment[j, k]`` = what_k of line 4;
    ``poorly_represented[j]`` = 1{max_k what_k(X~_j) <= t_n_eff}, the
    empirical counterpart of the Lemma B.2 bad event whose probability is the
    third term of Theorem 3.3 (population mu_k replaced by the train-sample
    averages the algorithm itself uses -- a diagnostic flag, not the theorem
    event); ``qhat[j]`` = +inf marks a vacuous prediction set.
    """

    qhat: Array
    selected: NDArray[np.int64]
    alignment: Array
    alignment_threshold: float
    poorly_represented: NDArray[np.bool_]
    w_inf: Array
    x_tilde: Array
    covered: Array | None
    n_vacuous: int


def ms_rlcp_predict(
    sources: Sequence[SourceCalibration],
    x_test: Array,
    *,
    alpha: float,
    kernel: LocalizationKernel,
    test_scores: Array | None = None,
    rng: RNG | None = None,
    x_tilde: Array | None = None,
) -> MSRLCPPrediction:
    """Multi-Source RLCP prediction sets (MS paper Alg. 1).

    Steps: draw X~_j ~ H(X_j, .) (line 1); per source k compute the alignment
    what_k = mean over ``x_train`` of H(X, X~_j) (line 4); select k_hat =
    argmax_k what_k (line 6, ties lowest index); on the selected source's
    calibration rows form Z_j = H(X_j, X~_j) + sum_i H(X_i, X~_j), weights
    w_i = H(X_i, X~_j) / Z_j and the +inf atom w_{n+1} = H(X_j, X~_j) / Z_j
    (lines 8-9); qhat_j is the (1 - alpha)-quantile of the weighted empirical
    distribution (line 10). The prediction set is {y : s_{k_hat}(X_j, y) <=
    qhat_j} (line 11); ``test_scores`` must therefore be evaluated under the
    *selected* source's score function when coverage is requested.
    """
    if len(sources) == 0:
        raise ValueError("at least one source is required")
    a = _validate_alpha(alpha)
    xt = _as_2d(x_test, "x_test")
    for src in sources:
        if src.x_train.shape[1] != xt.shape[1] or src.x_cal.shape[1] != xt.shape[1]:
            raise ValueError("all sources and x_test must share the feature dimension")
        if src.x_train.shape[1] != kernel.dim:
            raise ValueError(f"features must have dim {kernel.dim} matching the kernel")
    x_til = _resolve_x_tilde(kernel, xt, rng, x_tilde)
    m, k_n = xt.shape[0], len(sources)
    # Alg. 1 lines 4-6: alignment scores and argmax source selection.
    alignment = np.column_stack(
        [kernel.h_matrix(x_til, src.x_train).mean(axis=1) for src in sources]
    )
    alignment = np.asarray(alignment, dtype=float)
    selected = np.asarray(np.argmax(alignment, axis=1), dtype=np.int64)
    n_eff = min(int(src.x_train.shape[0]) for src in sources)
    t = alignment_threshold(kernel.h_sup, k_n, n_eff)
    poorly = np.asarray(alignment.max(axis=1) <= t, dtype=bool)
    # Alg. 1 lines 8-10: local weighted calibration on the selected source.
    h_test = kernel.h_rows(xt, x_til)
    qhat = np.full(m, np.inf, dtype=float)
    w_inf = np.zeros(m, dtype=float)
    for k in range(k_n):
        rows = np.flatnonzero(selected == k)
        if rows.size == 0:
            continue
        src = sources[k]
        h_cal = kernel.h_matrix(x_til[rows], src.x_cal)
        z = h_test[rows] + h_cal.sum(axis=1)
        if not bool(np.all(np.isfinite(z))) or bool(np.any(z <= 0.0)):
            raise ValueError(
                "kernel weights vanish (Z <= 0) at the selected source: the "
                "weighted score distribution is undefined -- widen the bandwidth"
            )
        w = np.asarray(h_cal / z[:, None], dtype=float)
        wi = np.asarray(h_test[rows] / z, dtype=float)
        qhat[rows] = _quantile_rows(src.cal_scores, w, wi, a)
        w_inf[rows] = wi
    covered: Array | None = None
    if test_scores is not None:
        ts = _finite_1d(test_scores, "test_scores")
        if ts.size != m:
            raise ValueError("test_scores must have one value per test row")
        covered = np.asarray(ts <= qhat, dtype=float)
    return MSRLCPPrediction(
        qhat=qhat,
        selected=selected,
        alignment=alignment,
        alignment_threshold=t,
        poorly_represented=poorly,
        w_inf=w_inf,
        x_tilde=x_til,
        covered=covered,
        n_vacuous=int(np.count_nonzero(~np.isfinite(qhat))),
    )


@dataclass(frozen=True)
class EnvelopeConstants:
    """Envelope quantities of MS paper Def. 1 / Lemma 3.1 (grid plug-ins).

    ``g_source_sup`` estimates max_k ||g_k||_inf, which equals B exactly
    (Lemma 3.1); the grid estimate approaches B from below as the grid refines.
    """

    b_envelope: float
    g_test_sup: float
    g_source_sup: float
    lipschitz: float
    n_grid: int


def envelope_constants_1d(
    source_densities: Sequence[DensityFn],
    test_density: DensityFn,
    *,
    lo: float,
    hi: float,
    n_grid: int = 4001,
) -> EnvelopeConstants:
    """Oracle SYNTHETIC envelope constants for X = R (MS paper Def. 1).

    Grid plug-ins on ``linspace(lo, hi, n_grid)``: B = int max_k f_k dx
    (trapezoid); g_test = B f_test / max_k f_k (sup over grid cells where the
    envelope is positive); ``lipschitz`` is the largest grid-slope estimate
    max |g(x + dx) - g(x)| / dx over {g_1..g_K, g_test}. These feed
    :func:`envelope_coverage_bound`, whose Theorem 3.3 statement needs the
    *true* constants of the true densities -- the estimates are only as good
    as the supplied (oracle, synthetic) densities and the grid resolution, so
    this is a correctness diagnostic, never market evidence. Fail-closed when
    the test density carries mass where every source density vanishes
    (Assumption 1 violated) or on degenerate grids/densities.
    """
    if len(source_densities) == 0:
        raise ValueError("source_densities must be non-empty")
    a, b = float(lo), float(hi)
    if not np.isfinite(a) or not np.isfinite(b) or a >= b:
        raise ValueError("require finite lo < hi")
    ng = int(n_grid)
    if ng < 3:
        raise ValueError("n_grid must be >= 3")
    grid = np.linspace(a, b, ng)
    dx = (b - a) / (ng - 1)

    def _eval(fn: DensityFn, name: str) -> Array:
        v = np.asarray(fn(grid), dtype=float).reshape(-1)
        if v.size != ng:
            raise ValueError(f"{name} must return one value per grid point")
        if not bool(np.all(np.isfinite(v))):
            raise ValueError(f"{name} must be finite on the grid")
        if bool(np.any(v < 0.0)):
            raise ValueError(f"{name} must be non-negative")
        return v

    fs = [_eval(fn, f"source_densities[{k}]") for k, fn in enumerate(source_densities)]
    f_test = _eval(test_density, "test_density")
    env = np.max(np.vstack(fs), axis=0)
    mask = env > 0.0
    if not bool(np.any(mask)):
        raise ValueError("all source densities vanish on the grid")
    if bool(np.any(~mask)) and float(np.max(f_test[~mask])) > 0.0:
        raise ValueError(
            "Assumption 1 violated: test density is positive where every source "
            "density vanishes (no distribution-free information there)"
        )
    b_env = float(np.sum((env[:-1] + env[1:]) * 0.5) * dx)
    if not np.isfinite(b_env) or b_env <= 0.0:
        raise ValueError("envelope normalizing constant B must be positive")

    def _ratio(f: Array) -> Array:
        g = np.full(ng, np.nan)
        g[mask] = b_env * f[mask] / env[mask]
        return g

    def _slope(g: Array) -> float:
        pairs = mask[:-1] & mask[1:]
        if not bool(np.any(pairs)):
            return 0.0
        return float(np.max(np.abs(np.diff(g)[pairs])) / dx)

    gs = [_ratio(f) for f in fs]
    g_test = _ratio(f_test)
    g_source_sup = float(max(float(np.nanmax(g)) for g in gs))
    g_test_sup = float(np.nanmax(g_test))
    lipschitz = float(max([_slope(g) for g in gs] + [_slope(g_test)]))
    return EnvelopeConstants(
        b_envelope=b_env,
        g_test_sup=g_test_sup,
        g_source_sup=g_source_sup,
        lipschitz=lipschitz,
        n_grid=ng,
    )


@dataclass(frozen=True)
class EnvelopeBound:
    """Theorem 3.3 coverage lower bound, decomposed (MS paper Sec. 3.1).

    ``bound`` = nominal - finite_sample_term - localization_term -
    representation_term; ``vacuous`` = bound <= 0 (flagged, never hidden).
    Term meanings: (a) finite_sample_term = 1 / n_eff; (b) localization_term
    = 2 L (1 + 2 ||g_test||_inf / B) E||X - X~||_2, the O(h) perturbation
    cost; (c) representation_term = P(max_k mu_k(X~) <= t_n_eff), how often
    the perturbed test feature lands where no source has local mass.
    """

    bound: float
    nominal: float
    finite_sample_term: float
    localization_term: float
    representation_term: float
    alignment_threshold: float
    mean_perturb_distance: float
    b_envelope: float
    g_test_sup: float
    lipschitz: float
    n_eff: int
    n_sources: int
    vacuous: bool


def envelope_coverage_bound(
    *,
    alpha: float,
    kernel: LocalizationKernel,
    lipschitz: float,
    g_test_sup: float,
    b_envelope: float,
    x_test: Array,
    source_train: Sequence[Array],
    rng: RNG | None = None,
    x_tilde: Array | None = None,
) -> EnvelopeBound:
    """Finite-sample Theorem 3.3 lower bound as a PLUG-IN diagnostic.

        P(Y in C) >= (1 - alpha - 1/n_eff)
                     - 2 L (1 + 2 ||g_test||_inf / B) E||X - X~||_2
                     - P_{P_test}(max_k mu_k(X~) <= t_n_eff)

    with n_eff = min_k n_{k,train} derived from ``source_train``, and
    t_n_eff = 2 ||H||_inf sqrt(2 log(2 K n_eff) / n_eff). ``lipschitz`` (L of
    the envelope-ratios g_k, g_test), ``g_test_sup`` and ``b_envelope`` are
    oracle constants (e.g. from :func:`envelope_constants_1d` on true
    SYNTHETIC densities); E||X - X~||_2 is the exact kernel constant. The
    representation term is a seeded Monte Carlo plug-in: mu_k(X~) is replaced
    by the train-sample average of Alg. 1 line 4 over ``source_train[k]`` and
    the probability by the mean over ``x_test`` (perturbed by the same sampler
    as the prediction path). With estimated constants the returned number is a
    diagnostic, not a theorem certificate -- and a value <= 0 is flagged
    ``vacuous`` rather than silently floored. Fail-closed on n_eff < 2,
    non-finite/negative constants, or shape mismatches.
    """
    a = _validate_alpha(alpha)
    if len(source_train) == 0:
        raise ValueError("source_train must be non-empty")
    trains = [_as_2d(s, f"source_train[{k}]") for k, s in enumerate(source_train)]
    xt = _as_2d(x_test, "x_test")
    for tr in trains:
        if tr.shape[1] != xt.shape[1]:
            raise ValueError("source_train and x_test must share the feature dimension")
    if xt.shape[1] != kernel.dim:
        raise ValueError(f"features must have dim {kernel.dim} matching the kernel")
    lip = float(lipschitz)
    if not np.isfinite(lip) or lip < 0.0:
        raise ValueError("lipschitz must be finite and >= 0")
    g_sup = float(g_test_sup)
    if not np.isfinite(g_sup) or g_sup < 0.0:
        raise ValueError("g_test_sup must be finite and >= 0")
    b_env = float(b_envelope)
    if not np.isfinite(b_env) or b_env <= 0.0:
        raise ValueError("b_envelope must be finite and > 0")
    k_n = len(trains)
    n_eff = min(int(tr.shape[0]) for tr in trains)
    t = alignment_threshold(kernel.h_sup, k_n, n_eff)
    x_til = _resolve_x_tilde(kernel, xt, rng, x_tilde)
    align = np.column_stack([kernel.h_matrix(x_til, tr).mean(axis=1) for tr in trains])
    rep = float(np.mean(np.asarray(align, dtype=float).max(axis=1) <= t))
    e_dist = kernel.mean_perturb_distance
    loc = float(2.0 * lip * (1.0 + 2.0 * g_sup / b_env) * e_dist)
    finite_term = 1.0 / float(n_eff)
    bound = (1.0 - a - finite_term) - loc - rep
    return EnvelopeBound(
        bound=float(bound),
        nominal=1.0 - a,
        finite_sample_term=float(finite_term),
        localization_term=loc,
        representation_term=rep,
        alignment_threshold=t,
        mean_perturb_distance=e_dist,
        b_envelope=b_env,
        g_test_sup=g_sup,
        lipschitz=lip,
        n_eff=n_eff,
        n_sources=k_n,
        vacuous=bool(bound <= 0.0),
    )


# ---------------------------------------------------------------------------
# SYNTHETIC validation: seeded two-bump DGP. Correctness material only --
# never market evidence.
# ---------------------------------------------------------------------------

_SIGMA_BASE = 0.4
_SIGMA_SLOPE = 0.22


def _bump_sigma(x: Array) -> Array:
    """Shared conditional scale sigma(x) = 0.4 + 0.22 |x| (covariate shift only)."""
    return np.asarray(_SIGMA_BASE + _SIGMA_SLOPE * np.abs(np.asarray(x, dtype=float)), dtype=float)


def _gaussian_pdf(x: Array, mu: float, sd: float) -> Array:
    z = (np.asarray(x, dtype=float) - mu) / sd
    return np.asarray(np.exp(-0.5 * z * z) / (sd * math.sqrt(2.0 * math.pi)), dtype=float)


def bench_ms_rlcp_two_bumps(
    *,
    seed: int = 2609,
    alpha: float = 0.10,
    n_train: int = 400,
    n_cal: int = 400,
    n_test_in: int = 600,
    n_test_gap: int = 300,
    bandwidth: float = 0.75,
    bump_std: float = 0.7,
    bump_sep: float = 3.0,
    gap_std: float = 0.5,
    n_grid: int = 4001,
) -> dict[str, float | str]:
    """SYNTHETIC MS-RLCP validation on two disjoint-ish Gaussian bumps.

    DGP (seeded, no market data): source A has X ~ N(-sep, std^2), source B
    X ~ N(+sep, std^2); the shared conditional is Y = sigma(X) eps,
    eps ~ N(0, 1), sigma(x) = 0.4 + 0.22 |x| (covariate shift only, so the MS
    paper's shared-P_{Y|X} assumption holds exactly). Each source fits the
    constant mean m_k on its train split and scores s_k(x, y) = |y - m_k|
    (FitScore external, Alg. 1 line 3). In-source test features come from the
    50/50 source mixture (the classical mixture regime); gap test features
    come from N(0, gap_std^2) -- a region thinly covered by both sources,
    absolutely continuous w.r.t. the Gaussian envelope (Assumption 1 holds)
    but with a huge envelope ratio ||g_test||_inf, exactly where Theorem 3.3
    must degrade visibly instead of hiding it.

    Returns research metrics only (proper calibration scores: coverage,
    widths, vacuity and poor-representation flag rates, Theorem 3.3 bound
    terms). The bound constants come from :func:`envelope_constants_1d` on
    the true bump densities (oracle grid plug-in). Correctness material;
    never market evidence, never a Sharpe-family headline.
    """
    a = _validate_alpha(alpha)
    nt, nc = int(n_train), int(n_cal)
    nti, ntg = int(n_test_in), int(n_test_gap)
    if min(nt, nc, nti, ntg) < 2:
        raise ValueError("n_train, n_cal, n_test_in, n_test_gap must be >= 2")
    bs, bsep, gs = float(bump_std), float(bump_sep), float(gap_std)
    if not np.isfinite(bs) or bs <= 0.0:
        raise ValueError("bump_std must be finite and > 0")
    if not np.isfinite(bsep) or bsep <= 0.0:
        raise ValueError("bump_sep must be finite and > 0")
    if not np.isfinite(gs) or gs <= 0.0:
        raise ValueError("gap_std must be finite and > 0")
    kernel = LocalizationKernel(kind="gaussian", bandwidth=float(bandwidth), dim=1)

    rng = np.random.default_rng(int(seed))

    def _draw_source(mu: float, n: int) -> tuple[Array, Array]:
        x = mu + bs * rng.standard_normal(n)
        y = _bump_sigma(x) * rng.standard_normal(n)
        return np.asarray(x, dtype=float), np.asarray(y, dtype=float)

    x_tr_a, y_tr_a = _draw_source(-bsep, nt)
    x_tr_b, y_tr_b = _draw_source(bsep, nt)
    x_cal_a, y_cal_a = _draw_source(-bsep, nc)
    x_cal_b, y_cal_b = _draw_source(bsep, nc)
    m_a = float(np.mean(y_tr_a))
    m_b = float(np.mean(y_tr_b))
    sources = [
        SourceCalibration(x_tr_a, x_cal_a, np.abs(y_cal_a - m_a)),
        SourceCalibration(x_tr_b, x_cal_b, np.abs(y_cal_b - m_b)),
    ]

    side = rng.random(nti) < 0.5
    x_in = np.where(side, -bsep, bsep) + bs * rng.standard_normal(nti)
    y_in = _bump_sigma(x_in) * rng.standard_normal(nti)
    x_gap = gs * rng.standard_normal(ntg)
    y_gap = _bump_sigma(x_gap) * rng.standard_normal(ntg)

    def _evaluate(x_te: Array, y_te: Array) -> tuple[MSRLCPPrediction, Array, Array]:
        pred = ms_rlcp_predict(sources, x_te, alpha=a, kernel=kernel, rng=rng)
        m_sel = np.where(pred.selected == 0, m_a, m_b)
        s_te = np.asarray(np.abs(y_te - m_sel), dtype=float)
        cov = np.asarray(s_te <= pred.qhat, dtype=float)  # +inf qhat covers
        widths = 2.0 * pred.qhat
        return pred, cov, widths

    pred_in, cov_in, width_in = _evaluate(x_in, y_in)
    pred_gap, cov_gap, width_gap = _evaluate(x_gap, y_gap)
    fin_in = np.isfinite(pred_in.qhat)
    fin_gap = np.isfinite(pred_gap.qhat)

    def _f_a(x: Array) -> Array:
        return _gaussian_pdf(x, -bsep, bs)

    def _f_b(x: Array) -> Array:
        return _gaussian_pdf(x, bsep, bs)

    def _f_test_in(x: Array) -> Array:
        return np.asarray(0.5 * (_f_a(x) + _f_b(x)), dtype=float)

    def _f_test_gap(x: Array) -> Array:
        return _gaussian_pdf(x, 0.0, gs)

    span = bsep + 8.0 * max(bs, gs)
    env_in = envelope_constants_1d([_f_a, _f_b], _f_test_in, lo=-span, hi=span, n_grid=n_grid)
    env_gap = envelope_constants_1d([_f_a, _f_b], _f_test_gap, lo=-span, hi=span, n_grid=n_grid)
    bound_in = envelope_coverage_bound(
        alpha=a,
        kernel=kernel,
        lipschitz=env_in.lipschitz,
        g_test_sup=env_in.g_test_sup,
        b_envelope=env_in.b_envelope,
        x_test=x_in,
        source_train=[x_tr_a, x_tr_b],
        rng=rng,
    )
    bound_gap = envelope_coverage_bound(
        alpha=a,
        kernel=kernel,
        lipschitz=env_gap.lipschitz,
        g_test_sup=env_gap.g_test_sup,
        b_envelope=env_gap.b_envelope,
        x_test=x_gap,
        source_train=[x_tr_a, x_tr_b],
        rng=rng,
    )

    region_a = side
    region_b = ~side
    return {
        "synthetic_dgp": "synthetic_two_bumps",
        "synthetic_claim": "research_metric_only",
        "synthetic_kernel": "gaussian",
        "synthetic_seed": float(seed),
        "synthetic_alpha": a,
        "synthetic_bandwidth": float(bandwidth),
        "synthetic_bump_std": bs,
        "synthetic_bump_sep": bsep,
        "synthetic_gap_std": gs,
        "synthetic_n_sources": 2.0,
        "synthetic_n_train": float(nt),
        "synthetic_n_cal": float(nc),
        "synthetic_n_eff": float(bound_in.n_eff),
        "synthetic_alignment_threshold": float(bound_in.alignment_threshold),
        "synthetic_n_test_in": float(nti),
        "synthetic_n_test_gap": float(ntg),
        "synthetic_coverage_in_source": float(np.mean(cov_in)),
        "synthetic_coverage_in_source_region_a": float(np.mean(cov_in[region_a])),
        "synthetic_coverage_in_source_region_b": float(np.mean(cov_in[region_b])),
        "synthetic_mean_width_in_source": float(np.mean(width_in[fin_in])),
        "synthetic_qhat_median_in_source": float(np.median(pred_in.qhat[fin_in])),
        "synthetic_frac_vacuous_in_source": float(np.mean(~fin_in)),
        "synthetic_poorly_represented_rate_in_source": float(np.mean(pred_in.poorly_represented)),
        "synthetic_coverage_gap": float(np.mean(cov_gap)),
        "synthetic_mean_width_gap_finite": float(np.mean(width_gap[fin_gap])),
        "synthetic_qhat_median_gap_finite": float(np.median(pred_gap.qhat[fin_gap])),
        "synthetic_frac_vacuous_gap": float(np.mean(~fin_gap)),
        "synthetic_poorly_represented_rate_gap": float(np.mean(pred_gap.poorly_represented)),
        "synthetic_selection_rate_a_region_a": float(np.mean(pred_in.selected[region_a] == 0)),
        "synthetic_selection_rate_b_region_b": float(np.mean(pred_in.selected[region_b] == 1)),
        "synthetic_selection_rate_a_gap": float(np.mean(pred_gap.selected == 0)),
        "synthetic_envelope_b": float(env_in.b_envelope),
        "synthetic_envelope_g_source_sup": float(env_in.g_source_sup),
        "synthetic_envelope_g_test_sup_in_source": float(env_in.g_test_sup),
        "synthetic_envelope_g_test_sup_gap": float(env_gap.g_test_sup),
        "synthetic_envelope_lipschitz_in_source": float(env_in.lipschitz),
        "synthetic_envelope_lipschitz_gap": float(env_gap.lipschitz),
        "synthetic_mean_perturb_distance": float(bound_in.mean_perturb_distance),
        "synthetic_localization_term_in_source": float(bound_in.localization_term),
        "synthetic_localization_term_gap": float(bound_gap.localization_term),
        "synthetic_representation_term_in_source": float(bound_in.representation_term),
        "synthetic_representation_term_gap": float(bound_gap.representation_term),
        "synthetic_bound_in_source": float(bound_in.bound),
        "synthetic_bound_gap": float(bound_gap.bound),
        "synthetic_bound_vacuous_in_source": float(bound_in.vacuous),
        "synthetic_bound_vacuous_gap": float(bound_gap.vacuous),
    }
