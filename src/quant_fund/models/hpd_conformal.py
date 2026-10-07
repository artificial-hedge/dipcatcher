"""C-USIM: HPD split conformal for possibly multimodal predictive laws. No Sharpe.

Implements the Conditionally-Uniformized Score Integration Method (C-USIM) of
Park, Park & Chang (2026), "Conformal Prediction and Conditional Coverage for
Tabular Foundation Models", arXiv:2609.34887 [stat.ML]. C-USIM is HPD-split
conformal prediction (Izbicki, Shimizu & Stern 2022, JMLR 23(87)) applied to
any predictive distribution available as a sample cloud and/or a density
evaluator -- e.g. tabular-foundation-model outputs reconstructed into
piecewise-uniform densities (paper App. B.1) or Monte Carlo posteriors. It
requires no additional training, model inference, or separate density /
score-correction model.

Nonconformity score (paper Sec. 4.2), the model-based PIT of the negated
predictive density:

    s_M(x, y) = P_{Z ~ M(x)}( f_{M(x)}(y) <= f_{M(x)}(Z) )

the predictive mass of the region whose density is at least f(y) -- the
density-rank of y, equivalently the HPD level (Hyndman 1996) at which y
enters the predictive region. Small scores conform. Calibration (paper
Alg. 1) is split conformal (Vovk, Gammerman & Shafer 2005): the threshold
qhat is the k-th order statistic of the calibration scores augmented with
+inf, k = ceil((n_cal + 1) * (1 - alpha)), giving finite-sample marginal
validity P(Y in C(X)) >= 1 - alpha under exchangeability. The predicted
region {y : s(x, y) <= qhat} may be disjoint, so a bimodal predictive law no
longer forces a connected interval to over-cover the low-density gap between
the modes (paper Sec. 3 vs. CQR/CHR).

Also provided: the conditional-marginal coverage-gap bound of paper Thm. 1,

    B(x) = (1/2) * || p_tilde(x) - p_hat(x) ||_1 + max_i p_hat_i(x),
    Delta(x) <= E_X[B(X)] + B(x),

split into a distribution-estimation term (half the L1 distance between the
true and predicted bin-probability vectors) and a score-discreteness term
(the largest predicted bin probability), and the percentile rank-score
diagnostic of paper Sec. 5.2 / App. B.3, which estimates the conditional
coverage P_t(x) = P(s(x, Y) <= t | X = x) per input from Monte Carlo draws
of the true conditional law. Under the paper's ideal case (Assumptions 2-3)
the conditional coverage follows Beta(k, n_cal + 1 - k) (paper Thm. 2).

Honesty: marginal validity never implies conditional coverage --
distribution-free conditional guarantees are impossible in finite samples
(Barber, Candes, Ramdas & Tibshirani 2021). The gap bound and the rank-score
diagnostic need the true conditional law, so they are oracle diagnostics for
SYNTHETIC correctness tests, never market evidence. bench_cusim_bimodal runs
on seeded simulated mixtures only. Coverage / set-length are proper
calibration metrics; no Sharpe-family headline is produced here.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.stats import gaussian_kde, norm

from quant_fund.metrics.conformal import conformal_quantile

Array = NDArray[np.float64]
DensityFn = Callable[[Array], Array]

#: Tolerance absorbing float representation error in ``score <= threshold``
#: comparisons. HPD scores are integer multiples of ``1 / m``, so the slack
#: never moves a genuinely interior or exterior point.
SCORE_TOL = 1e-12

#: Default factor turning the cloud's median nearest-neighbour spacing into
#: the gap tolerance that splits 1-D region members into interval components.
#: A rendering heuristic for the interval hull only -- the member mask (and
#: therefore coverage) is exact at any tolerance.
MERGE_TOL_FACTOR = 32.0

#: Probability vectors (paper Thm. 1 bins) must sum to 1 within this slack.
PROB_SUM_TOL = 1e-6


@dataclass(frozen=True)
class HPDRegion:
    """1-D representation of ``{y : s(x, y) <= threshold}`` (paper Alg. 1).

    ``members`` are the predictive-sample points inside the region;
    ``intervals`` are the interval hulls of maximal consecutive member runs
    (components), so a multimodal region yields several disjoint intervals.
    ``total_length`` sums component lengths and excludes the gaps, matching
    the prediction-set length measure of paper App. B.4.
    """

    members: Array
    intervals: Array
    total_length: float
    n_components: int
    threshold: float
    n_members: int


@dataclass(frozen=True)
class CUSIMCalibration:
    """Split-conformal threshold on HPD scores (paper Alg. 1, lines 10-11)."""

    qhat: float
    alpha: float
    n_cal: int
    k: int
    scores: Array


@dataclass(frozen=True)
class CUSIMPrediction:
    """HPD prediction region for one input, plus optional point coverage."""

    region: HPDRegion
    score: float | None
    covered: bool | None


@dataclass(frozen=True)
class GapBound:
    """Paper Thm. 1 bound ``Delta(x) <= E_X[B(X)] + B(x)`` and its terms."""

    est_error_x: float
    discreteness_x: float
    b_x: float
    expected_b: float
    bound: float
    n_pop: int


@dataclass(frozen=True)
class RankScoreTable:
    """Percentile rank-score diagnostic table (paper Sec. 5.2 / App. B.3)."""

    conditional_coverage: Array
    percentile_ranks: Array | None
    threshold: float
    target: float
    summary: dict[str, float]


def _finite_1d(values: object, name: str) -> Array:
    v = np.asarray(values, dtype=float).reshape(-1)
    if v.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if not bool(np.all(np.isfinite(v))):
        raise ValueError(f"{name} must be finite")
    return v


def default_kde_density(samples: Array) -> DensityFn:
    """Deterministic Gaussian KDE (Scott bandwidth) over the sample cloud.

    Fail-closed on degenerate clouds: a KDE needs at least two distinct
    sample values (zero variance makes the bandwidth matrix singular).
    """
    z = _finite_1d(samples, "samples")
    if int(np.unique(z).size) < 2:
        raise ValueError(
            "default Gaussian KDE needs >= 2 distinct sample values; "
            "pass density=... or sample_densities=... explicitly"
        )
    kde = gaussian_kde(z)

    def _density(values: Array) -> Array:
        return np.asarray(kde(values), dtype=float)

    return _density


class HPDScorer:
    """Sorted-density-rank HPD scorer for one predictive sample cloud.

    Density levels of the cloud are evaluated and sorted once, then any
    observation is scored by binary search -- the O(m log m) preprocessing /
    O(log m) query cost of paper App. B.5, which "sorts density levels once
    and computes cumulative probability masses, combining all intervals with
    equal density into the same score level". Equal densities therefore share
    one score level exactly (the ``>=`` in the score definition keeps all
    boundary ties, cf. paper Sec. 2.2).

    Parameters
    ----------
    samples:
        1-D predictive sample cloud ``Z_1..Z_m`` from ``M(x)``.
    density:
        Evaluator for ``f_{M(x)}`` applied to 1-D arrays. Default: Gaussian
        KDE over ``samples`` (``default_kde_density``). The score only depends
        on the density *ordering*, so unnormalized evaluators are valid.
    sample_densities:
        Optional precomputed ``f(Z_j)`` (paper App. B.1 binned/quantile
        reconstructions). When given without ``density``, only sample members
        can be scored; external observations fail closed.
    """

    def __init__(
        self,
        samples: Array,
        density: DensityFn | None = None,
        *,
        sample_densities: Array | None = None,
    ) -> None:
        z = _finite_1d(samples, "samples")
        self._samples = z
        self._m = int(z.size)
        resolved: DensityFn | None = density
        if sample_densities is None:
            if resolved is None:
                resolved = default_kde_density(z)
            d = np.asarray(resolved(z), dtype=float).reshape(-1)
        else:
            d = np.asarray(sample_densities, dtype=float).reshape(-1)
            if d.size != self._m:
                raise ValueError("sample_densities must have one value per sample")
        self._density = resolved
        if not bool(np.all(np.isfinite(d))):
            raise ValueError("density values on the sample cloud must be finite")
        if bool(np.any(d < 0.0)):
            raise ValueError("density values must be non-negative")
        self._d = d
        self._d_sorted = np.sort(d)
        # count_ge(d_j) = #{i : d_i >= d_j}: the density-rank of each sample.
        counts = self._m - np.searchsorted(self._d_sorted, d, side="left")
        self._sample_scores = np.asarray(counts / self._m, dtype=float)

    @property
    def n_samples(self) -> int:
        return self._m

    def score(self, y: Array | float) -> Array:
        """HPD nonconformity scores ``s(x, y)`` (paper Sec. 4.2), in [0, 1].

        ``s(x, y) = P_{Z ~ M(x)}(f(y) <= f(Z))`` estimated on the cloud as
        ``#{j : f(Z_j) >= f(y)} / m``. This is the HPD mass level at which y
        enters the predictive region: ``{y' : s(x, y') <= t}`` is the HPD
        region of predictive mass ``t``.
        """
        yy = _finite_1d(y, "observations y")
        if self._density is None:
            raise ValueError(
                "a density evaluator is required to score observations "
                "outside the sample cloud (sample_densities alone cannot)"
            )
        d_y = np.asarray(self._density(yy), dtype=float).reshape(-1)
        if d_y.size != yy.size:
            raise ValueError("density evaluator must return one value per observation")
        if not bool(np.all(np.isfinite(d_y))) or bool(np.any(d_y < 0.0)):
            raise ValueError("density evaluator must return finite non-negative values")
        counts = self._m - np.searchsorted(self._d_sorted, d_y, side="left")
        return np.asarray(counts / self._m, dtype=float)

    def region_mask(self, threshold: float) -> NDArray[np.bool_]:
        """Boolean mask of cloud members inside ``{y : s(x, y) <= threshold}``."""
        t = float(threshold)
        if not np.isfinite(t) or t < 0.0 or t > 1.0:
            raise ValueError("threshold must be finite and in [0, 1]")
        return self._sample_scores <= t + SCORE_TOL

    def region(self, threshold: float, *, merge_tol: float | None = None) -> HPDRegion:
        """HPD region at ``threshold``: members plus 1-D interval hulls.

        Components are maximal runs of sorted members whose consecutive gap
        is ``<= merge_tol``. Default ``merge_tol`` is ``MERGE_TOL_FACTOR``
        times the cloud's median nearest-neighbour spacing; pass an explicit
        value for degenerate clouds (fail-closed otherwise).
        """
        mask = self.region_mask(threshold)
        members = np.sort(self._samples[mask])
        t = float(threshold)
        if members.size == 0:
            return HPDRegion(
                members=members,
                intervals=np.empty((0, 2), dtype=float),
                total_length=0.0,
                n_components=0,
                threshold=t,
                n_members=0,
            )
        tol = self._resolve_merge_tol(merge_tol)
        splits = np.flatnonzero(np.diff(members) > tol)
        starts = np.concatenate(([0], splits + 1))
        ends = np.concatenate((splits + 1, [members.size]))
        intervals = np.column_stack((members[starts], members[ends - 1]))
        intervals = np.asarray(intervals, dtype=float)
        total = float(np.sum(intervals[:, 1] - intervals[:, 0]))
        return HPDRegion(
            members=members,
            intervals=intervals,
            total_length=total,
            n_components=int(intervals.shape[0]),
            threshold=t,
            n_members=int(members.size),
        )

    def _resolve_merge_tol(self, merge_tol: float | None) -> float:
        if merge_tol is not None:
            tol = float(merge_tol)
            if not np.isfinite(tol) or tol <= 0.0:
                raise ValueError("merge_tol must be positive and finite")
            return tol
        spacing = np.diff(np.sort(self._samples))
        med = float(np.median(spacing))
        if not np.isfinite(med) or med <= 0.0:
            raise ValueError(
                "cannot derive a merge tolerance from degenerate samples; pass merge_tol explicitly"
            )
        return MERGE_TOL_FACTOR * med


def hpd_score(
    samples: Array,
    y: Array | float,
    density: DensityFn | None = None,
    *,
    sample_densities: Array | None = None,
) -> Array:
    """HPD density-rank nonconformity scores of observations ``y``.

    Convenience wrapper around :class:`HPDScorer` computing
    ``s(x, y) = P_{Z ~ M(x)}(f_{M(x)}(y) <= f_{M(x)}(Z))`` (paper Sec. 4.2)
    on the predictive cloud ``samples`` -- the mass of the higher-density
    region, i.e. the HPD level at which ``y`` enters the predictive region.
    Returns one score per element of ``y``, each in ``[0, 1]``.
    """
    scorer = HPDScorer(samples, density, sample_densities=sample_densities)
    return scorer.score(y)


def cusim_calibrate(scores: Array, alpha: float) -> CUSIMCalibration:
    """Split-conformal threshold on HPD scores (paper Alg. 1, lines 10-11).

    ``qhat`` is the ``k``-th order statistic of the calibration scores
    augmented with ``+inf``, ``k = ceil((n_cal + 1) * (1 - alpha))``, computed
    by reusing :func:`quant_fund.metrics.conformal.conformal_quantile`.
    Finite-sample marginal validity ``P(Y in C(X)) >= 1 - alpha`` holds under
    exchangeability of calibration and test scores (paper Sec. 2.3, Assump. 2).

    Fail-closed: scores must be finite HPD masses in ``[0, 1]``; ``alpha``
    must lie in ``(0, 1)`` and be attainable, ``alpha >= 1 / (n_cal + 1)``
    (otherwise the paper's order statistic of ``{S_i} U {inf}`` is ``+inf``,
    i.e. the whole space -- we refuse instead of silently returning a
    vacuous region, mirroring ``conformal_quantile``'s caller contract).
    """
    s = _finite_1d(scores, "calibration scores")
    if bool(np.any(s < 0.0)) or bool(np.any(s > 1.0)):
        raise ValueError("HPD calibration scores are probability masses in [0, 1]")
    a = float(alpha)
    if not np.isfinite(a) or not 0.0 < a < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    n = int(s.size)
    k = int(np.ceil((n + 1) * (1.0 - a)))
    if k > n:
        raise ValueError(
            f"coverage level unattainable: alpha must be >= 1/(n_cal+1) = "
            f"{1.0 / (n + 1):.6g} for {n} calibration scores"
        )
    qhat = float(conformal_quantile(s, a))
    return CUSIMCalibration(qhat=qhat, alpha=a, n_cal=n, k=k, scores=s)


def cusim_predict(
    samples: Array,
    calibration: CUSIMCalibration,
    *,
    density: DensityFn | None = None,
    sample_densities: Array | None = None,
    y: float | None = None,
    merge_tol: float | None = None,
) -> CUSIMPrediction:
    """C-USIM prediction region ``{y' : s(x, y') <= qhat}`` (paper Alg. 1).

    Builds the HPD region at the calibrated threshold from the predictive
    cloud, represented by its sample members and 1-D interval hulls
    (:class:`HPDRegion`). When the observed response ``y`` is given, also
    returns its HPD score and the coverage flag ``score <= qhat`` -- the
    exact membership test of paper Alg. 1 line 12, which stays valid even
    when the interval hulls are only a coarse 1-D rendering of the region.
    """
    scorer = HPDScorer(samples, density, sample_densities=sample_densities)
    region = scorer.region(calibration.qhat, merge_tol=merge_tol)
    score: float | None = None
    covered: bool | None = None
    if y is not None:
        yv = float(y)
        if not np.isfinite(yv):
            raise ValueError("y must be finite")
        score = float(scorer.score(np.array([yv], dtype=float))[0])
        covered = bool(score <= calibration.qhat + SCORE_TOL)
    return CUSIMPrediction(region=region, score=score, covered=covered)


def _as_prob_row(values: object, name: str, width: int | None) -> Array:
    v = np.asarray(values, dtype=float)
    if v.ndim != 1 or v.size == 0:
        raise ValueError(f"{name} must be a non-empty 1-D probability vector")
    if width is not None and v.size != width:
        raise ValueError(f"{name} must have {width} bin probabilities")
    if not bool(np.all(np.isfinite(v))):
        raise ValueError(f"{name} must be finite")
    if bool(np.any(v < 0.0)):
        raise ValueError(f"{name} must be non-negative")
    if abs(float(v.sum()) - 1.0) > PROB_SUM_TOL:
        raise ValueError(f"{name} must sum to 1 within {PROB_SUM_TOL:g}")
    return v


def _b_row(p_hat: Array, p_tilde: Array) -> float:
    """Paper Thm. 1 term ``B(x) = (1/2)||p_tilde - p_hat||_1 + max_i p_hat_i``."""
    est = 0.5 * float(np.sum(np.abs(p_tilde - p_hat)))
    return est + float(np.max(p_hat))


def coverage_gap_bound(
    p_hat_x: Array,
    p_tilde_x: Array,
    *,
    p_hat_pop: Array | None = None,
    p_tilde_pop: Array | None = None,
) -> GapBound:
    """Conditional-marginal coverage-gap bound of paper Thm. 1.

    For a predictive law whose density-rank score is constant on the bins of
    a partition (the piecewise-uniform reconstruction of paper App. B.1) with
    distinct score values across bins:

        B(x)      = (1/2) * || p_tilde(x) - p_hat(x) ||_1 + max_i p_hat_i(x)
        Delta(x)  <= E_X[B(X)] + B(x)

    where ``p_hat_i(x)`` are predicted bin probabilities under ``M(x)`` and
    ``p_tilde_i(x)`` the true conditional bin probabilities ``P(Y in bin_i |
    X = x)``. The first term is distribution-estimation error, the second is
    score discreteness -- together they bound the gap between conditional and
    marginal coverage of the PIT-transformed (oracle conditional) score, via
    the KS-distance argument of paper App. A.1 (Laplante 2026).

    ``p_hat_pop`` / ``p_tilde_pop`` (rows over covariate draws, both required
    together) estimate ``E_X[B(X)]`` by their sample mean -- an oracle
    Monte Carlo quantity needing the true conditional law, so SYNTHETIC or
    simulation studies only. When omitted, the single vector at ``x`` serves
    as a one-draw proxy (``n_pop == 1``); the proxy is not the population
    expectation unless ``B`` is constant in ``X``.

    Finer partitions shrink the discreteness term but raise the L1
    estimation error (paper Sec. 5.1 trade-off). Fail-closed on vectors that
    are not finite, non-negative, equally wide, and summing to 1.
    """
    ph = _as_prob_row(p_hat_x, "p_hat_x", None)
    width = int(ph.size)
    pt = _as_prob_row(p_tilde_x, "p_tilde_x", width)
    est_error_x = 0.5 * float(np.sum(np.abs(pt - ph)))
    discreteness_x = float(np.max(ph))
    b_x = est_error_x + discreteness_x
    if (p_hat_pop is None) != (p_tilde_pop is None):
        raise ValueError("pass p_hat_pop and p_tilde_pop together (or neither)")
    if p_hat_pop is None or p_tilde_pop is None:
        return GapBound(
            est_error_x=est_error_x,
            discreteness_x=discreteness_x,
            b_x=b_x,
            expected_b=b_x,
            bound=2.0 * b_x,
            n_pop=1,
        )
    ph_pop = np.asarray(p_hat_pop, dtype=float)
    pt_pop = np.asarray(p_tilde_pop, dtype=float)
    if ph_pop.ndim != 2 or pt_pop.shape != ph_pop.shape:
        raise ValueError("p_hat_pop and p_tilde_pop must be matching 2-D arrays")
    if ph_pop.shape[1] != width or ph_pop.shape[0] == 0:
        raise ValueError(f"population arrays must have >= 1 row of {width} bins")
    b_pop = np.array(
        [
            _b_row(
                _as_prob_row(ph_pop[j], f"p_hat_pop[{j}]", width),
                _as_prob_row(pt_pop[j], f"p_tilde_pop[{j}]", width),
            )
            for j in range(ph_pop.shape[0])
        ]
    )
    expected_b = float(np.mean(b_pop))
    return GapBound(
        est_error_x=est_error_x,
        discreteness_x=discreteness_x,
        b_x=b_x,
        expected_b=expected_b,
        bound=expected_b + b_x,
        n_pop=int(ph_pop.shape[0]),
    )


def _quantile_summary(values: Array, prefix: str) -> dict[str, float]:
    q = np.quantile(values, [0.05, 0.25, 0.50, 0.75, 0.95])
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_std": float(np.std(values)),
        f"{prefix}_p05": float(q[0]),
        f"{prefix}_p25": float(q[1]),
        f"{prefix}_p50": float(q[2]),
        f"{prefix}_p75": float(q[3]),
        f"{prefix}_p95": float(q[4]),
        f"{prefix}_spread_p95_p05": float(q[4] - q[0]),
    }


def rank_score_percentiles(
    draw_scores: Array,
    threshold: float,
    *,
    target: float | None = None,
    observed_scores: Array | None = None,
) -> RankScoreTable:
    """Percentile rank-score diagnostic table (paper Sec. 5.2 / App. B.3).

    ``draw_scores[i, b]`` is ``T(X_i, Y~_i^(b))``: the HPD score, under the
    predictive law at input ``i``, of the ``b``-th Monte Carlo draw from the
    *true* conditional law ``Y | X = X_i``. The table reports

    * ``conditional_coverage[i]`` = ``P_t(X_i) = P(s <= t | X = X_i)``, the
      estimated conditional coverage at score threshold ``t`` -- the horizontal
      coordinate where input ``i``'s percentile rank-score curve crosses
      ``T = t`` in the paper's plots;
    * ``percentile_ranks[i]`` = ``P^(X_i, Y_i)`` of paper App. B.3, the
      conditional percentile rank of the observed score among the draws
      (needs ``observed_scores``); under the paper's Assumptions 2-3 these
      ranks are Unif(0, 1) when the predictive law is exact;
    * ``summary``: spread (p95 - p05), std, quartiles and
      ``ccad = mean_i |P_t(X_i) - target|`` (paper App. B.4). A miscalibrated
      predictor shows wider coverage spread and larger CCAD; ``target``
      defaults to ``threshold`` since HPD score levels are predictive masses.

    Requires the true conditional law for the draws, so it is an oracle
    SYNTHETIC diagnostic, never market evidence. Fail-closed on shapes,
    non-finite values, or scores outside ``[0, 1]``.
    """
    m = np.asarray(draw_scores, dtype=float)
    if m.ndim != 2 or m.size == 0:
        raise ValueError("draw_scores must be a non-empty (n_inputs, n_draws) matrix of HPD scores")
    if not bool(np.all(np.isfinite(m))):
        raise ValueError("draw_scores must be finite")
    if bool(np.any(m < 0.0)) or bool(np.any(m > 1.0)):
        raise ValueError("draw_scores must be HPD scores in [0, 1]")
    t = float(threshold)
    if not np.isfinite(t):
        raise ValueError("threshold must be finite")
    tgt = t if target is None else float(target)
    if not np.isfinite(tgt) or not 0.0 < tgt <= 1.0:
        raise ValueError("target must be in (0, 1]")
    cov = np.asarray(np.mean(m <= t + SCORE_TOL, axis=1), dtype=float)
    summary = _quantile_summary(cov, "cov")
    summary["ccad"] = float(np.mean(np.abs(cov - tgt)))
    summary["n_inputs"] = float(m.shape[0])
    summary["n_draws"] = float(m.shape[1])
    ranks: Array | None = None
    if observed_scores is not None:
        obs = _finite_1d(observed_scores, "observed_scores")
        if obs.size != m.shape[0]:
            raise ValueError("observed_scores must have one score per input row")
        if bool(np.any(obs < 0.0)) or bool(np.any(obs > 1.0)):
            raise ValueError("observed_scores must be HPD scores in [0, 1]")
        ranks = np.asarray(np.mean(m <= obs[:, None] + SCORE_TOL, axis=1), dtype=float)
        summary.update(_quantile_summary(ranks, "rank"))
    return RankScoreTable(
        conditional_coverage=cov,
        percentile_ranks=ranks,
        threshold=t,
        target=tgt,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# SYNTHETIC validation: seeded bimodal mixture DGP. Correctness material
# only -- never market evidence.
# ---------------------------------------------------------------------------

_SIGMA = 0.6
_MODE_HI = 3.0
_MODE_LO = -3.0
_MIS_MODE_LO = -2.0


def _gauss_pdf(y: Array, loc: float, sigma: float = _SIGMA) -> Array:
    z = (np.asarray(y, dtype=float) - loc) / sigma
    return np.asarray(np.exp(-0.5 * z * z) / (sigma * np.sqrt(2.0 * np.pi)), dtype=float)


def _true_density(w_hi: float) -> DensityFn:
    """Exact conditional density: w on the high mode, 1 - w on the low mode."""

    def _f(y: Array) -> Array:
        return np.asarray(
            w_hi * _gauss_pdf(y, _MODE_HI) + (1.0 - w_hi) * _gauss_pdf(y, _MODE_LO),
            dtype=float,
        )

    return _f


def _mis_density(y: Array) -> Array:
    """Miscalibrated predictor: 50/50 weights and the low mode at -2, not -3."""
    return np.asarray(
        0.5 * _gauss_pdf(y, _MODE_HI) + 0.5 * _gauss_pdf(y, _MIS_MODE_LO), dtype=float
    )


def _true_weight(x: Array) -> Array:
    """Covariate-dependent high-mode weight w(x) = 0.1 + 0.8 x on x in (0, 1)."""
    return np.asarray(0.1 + 0.8 * np.asarray(x, dtype=float), dtype=float)


def _draw_mixture(rng: np.random.Generator, w_hi: Array, n: int) -> Array:
    upper = rng.random(n) < np.asarray(w_hi, dtype=float)
    loc = np.where(upper, _MODE_HI, _MODE_LO)
    return np.asarray(loc + _SIGMA * rng.standard_normal(n), dtype=float)


def _draw_mixture_cloud(rng: np.random.Generator, w_hi: Array, n: int, m: int) -> Array:
    w = np.asarray(w_hi, dtype=float).reshape(-1, 1)
    upper = rng.random((n, m)) < w
    loc = np.where(upper, _MODE_HI, _MODE_LO)
    return np.asarray(loc + _SIGMA * rng.standard_normal((n, m)), dtype=float)


def _draw_mis_cloud(rng: np.random.Generator, m: int) -> Array:
    loc = np.where(rng.random(m) < 0.5, _MODE_HI, _MIS_MODE_LO)
    return np.asarray(loc + _SIGMA * rng.standard_normal(m), dtype=float)


def _bin_mass_matrix(edges: Array, w_hi: Array, loc_lo: float) -> Array:
    """Bin probabilities of ``w N(3, s^2) + (1-w) N(loc_lo, s^2)``.

    Rows cover ``(-inf, edges[0]]``, the interior bins, and ``[edges[-1], inf)``
    and sum to 1 by telescoping. Used for the paper Thm. 1 gap bound, whose
    piecewise-constant setting these fixed bins realize exactly.
    """
    hi_cdf = np.asarray(norm.cdf(edges, loc=_MODE_HI, scale=_SIGMA), dtype=float)
    lo_cdf = np.asarray(norm.cdf(edges, loc=loc_lo, scale=_SIGMA), dtype=float)
    w = np.asarray(w_hi, dtype=float).reshape(-1, 1)
    cdf = w * hi_cdf[None, :] + (1.0 - w) * lo_cdf[None, :]
    interior = np.diff(cdf, axis=1)
    return np.asarray(np.hstack([cdf[:, :1], interior, 1.0 - cdf[:, -1:]]), dtype=float)


def bench_cusim_bimodal(
    *,
    seed: int = 2609,
    alpha: float = 0.10,
    n_cal: int = 800,
    n_test: int = 1200,
    n_cloud: int = 400,
    n_diag: int = 120,
    n_draws: int = 2000,
) -> dict[str, float | str]:
    """SYNTHETIC C-USIM vs absolute-residual conformal on a bimodal DGP.

    DGP (seeded, no market data): ``X ~ Unif(0, 1)`` and
    ``Y | X = x ~ w(x) N(3, 0.6^2) + (1 - w(x)) N(-3, 0.6^2)`` with
    ``w(x) = 0.1 + 0.8 x``. The well-calibrated predictor uses the exact
    conditional density and conditional clouds; the miscalibrated predictor
    uses a fixed ``0.5 N(3, .) + 0.5 N(-2, .)`` mixture (wrong weight and
    wrong low mode). The baseline is split conformal on absolute residuals
    around the conditional mean ``mu(x) = 6 w(x) - 3``: its prediction set
    is one connected interval that must span the low-density gap between the
    modes, while the C-USIM HPD region (paper Alg. 1) may be disjoint.

    Returns research metrics only (proper calibration scores: coverage,
    set length, CCAD, rank-score spread, Thm. 1 gap-bound means). Correctness
    material; never market evidence, never a Sharpe-family headline.
    """
    a = float(alpha)
    if not np.isfinite(a) or not 0.0 < a < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    nc, nt, nclo, ndi, ndr = int(n_cal), int(n_test), int(n_cloud), int(n_diag), int(n_draws)
    if min(nc, nt, nclo, ndr) < 2 or ndi < 1:
        raise ValueError("n_cal, n_test, n_cloud, n_draws >= 2 and n_diag >= 1 required")
    if ndi > nt:
        raise ValueError("n_diag must not exceed n_test")
    if a < 1.0 / (nc + 1):
        raise ValueError("alpha too small for n_cal: level unattainable")

    rng = np.random.default_rng(int(seed))
    x_cal = rng.random(nc)
    w_cal = _true_weight(x_cal)
    y_cal = _draw_mixture(rng, w_cal, nc)
    clouds_cal = _draw_mixture_cloud(rng, w_cal, nc, nclo)
    scores_cal = np.array(
        [
            float(hpd_score(clouds_cal[i], y_cal[i], density=_true_density(w_cal[i]))[0])
            for i in range(nc)
        ]
    )
    calib = cusim_calibrate(scores_cal, a)

    mu_cal = _MODE_HI * w_cal + _MODE_LO * (1.0 - w_cal)
    q_abs = float(conformal_quantile(np.abs(y_cal - mu_cal), a))

    mis_scorer = HPDScorer(_draw_mis_cloud(rng, nclo), density=_mis_density)
    mis_calib = cusim_calibrate(mis_scorer.score(y_cal), a)

    x_test = rng.random(nt)
    w_test = _true_weight(x_test)
    y_test = _draw_mixture(rng, w_test, nt)
    clouds_test = _draw_mixture_cloud(rng, w_test, nt, nclo)
    lengths = np.empty(nt)
    ncomp = np.empty(nt)
    cov_c = np.empty(nt)
    scores_test = np.empty(nt)
    for i in range(nt):
        pred = cusim_predict(
            clouds_test[i], calib, density=_true_density(w_test[i]), y=float(y_test[i])
        )
        lengths[i] = pred.region.total_length
        ncomp[i] = float(pred.region.n_components)
        cov_c[i] = 1.0 if pred.covered else 0.0
        scores_test[i] = float(pred.score or 0.0)
    mu_test = _MODE_HI * w_test + _MODE_LO * (1.0 - w_test)
    cov_base = (np.abs(y_test - mu_test) <= q_abs + SCORE_TOL).astype(float)

    w_diag = w_test[:ndi]
    draws = _draw_mixture_cloud(rng, w_diag, ndi, ndr)
    well = np.empty((ndi, ndr))
    mis = np.empty((ndi, ndr))
    for i in range(ndi):
        scorer_i = HPDScorer(clouds_test[i], density=_true_density(w_diag[i]))
        well[i] = scorer_i.score(draws[i])
        mis[i] = mis_scorer.score(draws[i])
    obs_mis = mis_scorer.score(y_test[:ndi])
    tbl_well = rank_score_percentiles(
        well, calib.qhat, target=1.0 - a, observed_scores=scores_test[:ndi]
    )
    tbl_mis = rank_score_percentiles(mis, mis_calib.qhat, target=1.0 - a, observed_scores=obs_mis)

    edges = np.linspace(-8.0, 8.0, 65)
    p_true = _bin_mass_matrix(edges, w_diag, _MODE_LO)
    p_mis = _bin_mass_matrix(edges, np.full(ndi, 0.5), _MIS_MODE_LO)
    bound_well = np.array(
        [
            coverage_gap_bound(p_true[i], p_true[i], p_hat_pop=p_true, p_tilde_pop=p_true).bound
            for i in range(ndi)
        ]
    )
    bound_mis = np.array(
        [
            coverage_gap_bound(p_mis[i], p_true[i], p_hat_pop=p_mis, p_tilde_pop=p_true).bound
            for i in range(ndi)
        ]
    )

    mean_len_cusim = float(np.mean(lengths))
    mean_width_abs = float(2.0 * q_abs)
    return {
        "synthetic_dgp": "synthetic_bimodal_mixture",
        "synthetic_claim": "research_metric_only",
        "synthetic_seed": float(seed),
        "synthetic_alpha": a,
        "synthetic_n_cal": float(nc),
        "synthetic_n_test": float(nt),
        "synthetic_n_cloud": float(nclo),
        "synthetic_n_diag": float(ndi),
        "synthetic_n_draws": float(ndr),
        "synthetic_qhat_cusim": calib.qhat,
        "synthetic_qhat_miscal": mis_calib.qhat,
        "synthetic_qhat_absresid": q_abs,
        "synthetic_coverage_cusim": float(np.mean(cov_c)),
        "synthetic_coverage_absresid": float(np.mean(cov_base)),
        "synthetic_mean_total_length_cusim": mean_len_cusim,
        "synthetic_mean_width_absresid": mean_width_abs,
        "synthetic_length_ratio": mean_len_cusim / mean_width_abs
        if mean_width_abs > 0
        else float("nan"),
        "synthetic_n_components_mean": float(np.mean(ncomp)),
        "synthetic_coverage_spread_wellcal": tbl_well.summary["cov_spread_p95_p05"],
        "synthetic_coverage_spread_miscal": tbl_mis.summary["cov_spread_p95_p05"],
        "synthetic_coverage_std_wellcal": tbl_well.summary["cov_std"],
        "synthetic_coverage_std_miscal": tbl_mis.summary["cov_std"],
        "synthetic_ccad_wellcal": tbl_well.summary["ccad"],
        "synthetic_ccad_miscal": tbl_mis.summary["ccad"],
        "synthetic_gap_bound_wellcal_mean": float(np.mean(bound_well)),
        "synthetic_gap_bound_miscal_mean": float(np.mean(bound_mis)),
    }
