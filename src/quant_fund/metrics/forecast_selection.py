"""Target alignment, dilution and cautious forecast selection (common-target geometry).

Implements the exact, model-free geometry of Soleimani (2026), "Target
alignment, dilution and forecast selection when cross-sectional forecasts share
a common target", arXiv:2609.26303v1 [econ.EM], submitted 22 Sep 2026
(35 pages; journal reference: International Journal of Forecasting). Citation
verified against the arXiv abstract page and the v1 full text on 2026-09-30.

Setting (paper Section 2.1). At each date ``t`` every forecaster scores the
same ``M`` units (assets, products, regions); scores are standardized across
units and evaluated against one common standardized realized target ``y_t``
with the ``1/M`` inner product ``<a,b>_t = M^-1 a'b``, so ``||s_it||_t =
||y_t||_t = 1`` (paper Eq. (1)). The loss of a combination ``s_w = sum_i w_i
s_i`` is the relative-score loss ``||s_w,t - y_t||_t^2``; the zero forecast
(the *no-information forecast*) has loss exactly ``1``.

What is implemented (paper propositions verified by fetch):

- ``alignment_decomposition`` — Definition 1 + Proposition 1 (target-orthogonal
  decomposition): every standardized forecast splits exactly as
  ``s_it = gamma_it * y_t + u_it`` with ``gamma_it = <s_it, y_t>_t`` (the
  *target alignment*, a correlation because both vectors are unit-norm) and
  ``<u_it, y_t>_t = 0``, ``||u_it||_t^2 = 1 - gamma_it^2``,
  ``C_t = gamma_t gamma_t' + K_t`` with ``K_t`` PSD. Proposition 5: the
  common-target deviation correlation
  ``rho^e_ij,t = (1 + rho^s_ij,t - gamma_it - gamma_jt) /
  (2 sqrt((1-gamma_it)(1-gamma_jt)))`` — an affine translation of forecast
  correlation ``(1+rho^s)/2`` at zero alignment, hence a poor diversity
  measure.
- ``risk_decomposition`` — Proposition 2 (``R(w) = 1 - 2 g_w + q_w``),
  Proposition 3 (scale split ``R(w) = (1 - rho_w^2) + q_w (1 - c_w)^2``),
  Proposition 6 (pooled aggregation ``R(w) = (1-g_w)^2 + w' K_pool w`` with
  ``K_pool = K_bar + Cov_a(gamma_t)``), Corollary 1 (an equal-weight pool
  beats the no-information forecast iff mean alignment exceeds half the
  composite's squared norm; ``R(1/N) - 1 = -2m``).
- ``attainable_risk`` — Corollary 2: the best in-sample risk of any linear
  combination, ``1 - gamma_bar' C_bar^+ gamma_bar`` (a bound for the sample
  moments, not an operationally attainable out-of-sample risk).
- ``dilution_accounting`` — Propositions 8-9: the exact incremental risk of
  equal-weight admission, ``Delta_{A|P} = V_{P u A} - V_P``, split into the
  scale-free component ``Delta^sf = rho_P^2 - rho_{P u A}^2`` (genuine
  improvement: a lift of the pooled squared correlation) and the
  scale-mismatch component ``Delta^scale`` (mere dilution — per the paper's
  Discussion, "what the scale-mismatch component captures is dilution, not
  predictive content"). Includes the ``Lambda^EW`` descriptive ratio and the
  Proposition 9 dilution bound ``(m g_bar_A)^2 / (n^2 q_P)``.
- ``cautious_selection`` — the paper's three-way admission rule (Section 7):
  greedy selection starting from the highest-history-alignment forecaster;
  per-step simultaneous Newey-West HAC bands with a max-|t| critical value and
  a practical margin ``delta`` (admit iff ``Delta_hat + c*SE < -delta``,
  reject iff ``Delta_hat - c*SE > 0``, else undecided; interval-inclusion
  logic of equivalence testing, Schuirmann 1987). Bases: ``equal_weight``
  (raw composites) and ``scale_free`` (history-scale-calibrated composites,
  whose date-loss differences average exactly to ``Delta^sf``; plug-in scales
  affect aggregated risk only to second order, paper Appendix A). A
  ``point_estimate`` decision rule reproduces the paper's naive
  greedy-admission benchmark that can reward pure dilution.
- ``error_correlation_mirror`` — Section 9.2 diagnostics: deviation
  correlation versus the zero-alignment translation benchmark.
- ``planted_panel`` / ``forecast_selection_benchmarks`` — seeded SYNTHETIC
  validation world (k aligned + m mutually correlated zero-alignment diluters)
  and a flat ``dict[str, float]`` bench of proper diagnostics.

Guarantee (paper Remark 1, documented not strengthened): conditional on the
incumbent pool ``P`` and a fixed candidate family, if the date-level loss
differences satisfy a CLT and the HAC estimator is consistent, the
simultaneous bands cover all ``Delta_{A_j|P}`` with asymptotic probability
``1 - alpha``; the probability of admitting any candidate with
``Delta_{A_j|P} >= -delta`` is asymptotically bounded by ``alpha/2``,
*conditionally on P*. Greedy selection makes later incumbents
data-dependent, so no path-level (selective-inference) guarantee holds.

Honesty (AGENTS.md contract): research diagnostics only — relative-score risk
under the paper's estimand, IC-type correlations, admission statistics. No
Sharpe/Sortino/Calmar/P&L/NAV, no portfolio utility; the loss is not economic
value and "beating" the no-information forecast means improving standardized
relative-score risk only. SYNTHETIC panels are correctness fixtures, never
market evidence. Deterministic: all randomness is seeded; repeated calls are
bit-identical. Fail-closed: invalid or degenerate inputs raise; the paper's
exact algebraic identities are re-checked at runtime and a violation raises
rather than returning a tampered number.

Composes (does not modify) the Bates-Granger machinery:
``models.ensemble.equal_weights`` builds pool weight vectors;
``metrics.inference.newey_west_variance`` (Bartlett kernel, matching the
paper's Newey-West lag convention) supplies per-candidate HAC variances.

References:
- Soleimani (2026). Target alignment, dilution and forecast selection when
  cross-sectional forecasts share a common target. arXiv:2609.26303 [econ.EM].
- Bates, Granger (1969). The combination of forecasts. *OR Quarterly* 20.
- Newey, West (1987/1994) HAC; lag floor(4 (T/100)^(2/9)) per paper Section 7.
- Hothorn, Bretz, Westfall (2008); Romano, Wolf (2005) — max-|t| calibration.
- Schuirmann (1987) — equivalence-testing interval logic for the margin.
- Ledoit, Wolf (2004) — constant-correlation shrinkage context (Prop. 7 of the
  paper; not re-implemented here — admission decisions rely on realized losses).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import stats as sstats

from quant_fund.metrics.inference import newey_west_variance

Array = NDArray[np.float64]

__all__ = [
    "EPS_PSD",
    "EPS_SCALE",
    "MIN_SUPPORT",
    "NO_INFORMATION_RISK",
    "AlignmentDecomposition",
    "DilutionAccounting",
    "SelectionResult",
    "SelectionStep",
    "alignment_decomposition",
    "attainable_risk",
    "calibrated_composite_losses",
    "cautious_selection",
    "composite_losses",
    "composite_scores",
    "dilution_accounting",
    "error_correlation_mirror",
    "forecast_selection_benchmarks",
    "planted_panel",
    "relative_score_risk",
    "risk_decomposition",
]

# Paper Appendix B implementation settings.
EPS_SCALE = 1e-8  # epsilon_r = epsilon_x: degenerate dispersion tolerance
EPS_PSD = 1e-9  # epsilon_PSD
MIN_SUPPORT = 20  # minimum common support (units per date)
NO_INFORMATION_RISK = 1.0  # loss of the zero forecast, ||0 - y||_t^2 = 1

# Fail-closed tolerances for the paper's *exact* algebraic identities. The
# identities hold up to floating-point summation order only; a violation means
# a bug or tampered input, so we raise rather than report.
_IDENTITY_ATOL = 1e-8
_GAMMA_PERFECT_TOL = 1e-12
_Z_80 = float(sstats.norm.ppf(0.8))  # paper Section 7 minimum detectable effect


def _equal_weights(n: int) -> Array:
    """Delegate to ``models.ensemble.equal_weights`` (single source of truth).

    Lazy: models (analytics layer) sits above metrics (market_data layer).
    """
    from quant_fund.models.ensemble import equal_weights

    return equal_weights(n)


def _reject_bool_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int | np.integer):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _as_index_tuple(
    indices: Sequence[int] | NDArray[np.int64], n: int, name: str
) -> tuple[int, ...]:
    arr: Sequence[Any]
    if isinstance(indices, np.ndarray):
        if indices.ndim != 1:
            raise ValueError(f"{name} must be a 1-D index array")
        arr = indices.tolist()
    else:
        arr = list(indices)
    out: list[int] = []
    for value in arr:
        out.append(_reject_bool_int(value, name))
    if not out:
        raise ValueError(f"{name} must be nonempty")
    if any(i < 0 or i >= n for i in out):
        raise ValueError(f"{name} indices out of range for {n} forecasters")
    if len(set(out)) != len(out):
        raise ValueError(f"{name} must not repeat indices")
    return tuple(out)


def _as_weights(n_forecasters: int, weights: Array, name: str = "weights") -> Array:
    w = np.asarray(weights, dtype=float).reshape(-1)
    if w.size != n_forecasters or not np.all(np.isfinite(w)):
        raise ValueError(f"{name} must be a finite vector of length {n_forecasters}")
    return w


def _standardize_rows(x: Array, axis: int) -> Array:
    """Center/scale along ``axis`` with the paper's 1/M normalization (Eq. (1)).

    Uses the population (1/M) standard deviation so that standardized rows
    have unit norm under ``<a,b>_t = M^-1 a'b``. Degenerate dispersion must be
    screened out by the caller (fail-closed there, not here).
    """
    center = x.mean(axis=axis, keepdims=True)
    scale = x.std(axis=axis, keepdims=True)
    return np.asarray((x - center) / scale, dtype=float)


@dataclass(frozen=True)
class AlignmentDecomposition:
    """Exact target-aligned / target-orthogonal split (Definition 1, Prop. 1).

    Attributes hold only valid dates (rows where the target and every
    forecaster had non-degenerate dispersion; ``valid_mask`` maps back to the
    raw date axis). Shapes: ``scores``/``aligned``/``orthogonal`` are
    ``(T, N, M)``; ``target`` is ``(T, M)``; ``gamma`` is ``(T, N)``;
    ``forecast_corr``/``orthogonal_cov``/``deviation_corr`` are ``(T, N, N)``.
    """

    scores: Array  # standardized forecasts s_it (Eq. (1))
    target: Array  # standardized common target y_t (Eq. (1))
    gamma: Array  # target alignment gamma_it = <s_it, y_t>_t (Definition 1)
    aligned: Array  # gamma_it * y_t — projection onto the common target
    orthogonal: Array  # u_it = s_it - gamma_it * y_t, <u_it, y_t>_t = 0
    forecast_corr: Array  # C_t = [<s_it, s_jt>_t] (target-free dependence)
    orthogonal_cov: Array  # K_t = C_t - gamma_t gamma_t' (PSD, Prop. 1)
    deviation_corr: Array  # rho^e_ij,t common-target deviation corr. (Prop. 5)
    valid_mask: NDArray[np.bool_]  # raw dates that entered the analysis
    n_excluded_dates: int  # dates dropped for degenerate dispersion

    @property
    def n_dates(self) -> int:
        return int(self.scores.shape[0])

    @property
    def n_forecasters(self) -> int:
        return int(self.scores.shape[1])

    @property
    def n_units(self) -> int:
        return int(self.scores.shape[2])

    @property
    def mean_gamma(self) -> Array:
        """Time-averaged alignments gamma_bar (uniform weights a_t)."""
        return np.asarray(self.gamma.mean(axis=0), dtype=float)

    @property
    def mean_forecast_corr(self) -> Array:
        """Time-averaged forecast correlation C_bar."""
        return np.asarray(self.forecast_corr.mean(axis=0), dtype=float)

    @property
    def mean_orthogonal_cov(self) -> Array:
        """Time-averaged target-orthogonal covariance K_bar."""
        return np.asarray(self.orthogonal_cov.mean(axis=0), dtype=float)

    @property
    def mean_deviation_corr(self) -> Array:
        """Time-averaged common-target deviation correlation rho_bar^e (Prop. 5)."""
        return np.asarray(self.deviation_corr.mean(axis=0), dtype=float)

    @property
    def pooled_orthogonal_cov(self) -> Array:
        """K_pool = C_bar - gamma_bar gamma_bar' = K_bar + Cov_a(gamma_t) (Prop. 6)."""
        gbar = self.mean_gamma
        return np.asarray(self.mean_forecast_corr - np.outer(gbar, gbar), dtype=float)


def alignment_decomposition(
    scores: Array,
    target: Array,
    *,
    eps: float = EPS_SCALE,
    min_support: int = MIN_SUPPORT,
) -> AlignmentDecomposition:
    """Split every standardized forecast into aligned + target-orthogonal parts.

    Inputs are RAW score panels ``(T, N, M)`` (dates, forecasters, units) and
    raw realized targets ``(T, M)`` on the common support. Each date's scores
    and target are standardized across units with the paper's ``1/M``
    normalization (Eq. (1)), so ``<a,b>_t = M^-1 a'b`` and all norms are 1.
    (Already-standardized input is a fixed point of this step.) Dates whose
    target dispersion or any forecaster's dispersion is ``<= eps`` are
    excluded, matching the paper's fixed-membership rule (no imputation);
    ``valid_mask`` records which raw dates entered.

    The projection (Definition 1, Proposition 1): ``gamma_it = <s_it, y_t>_t``
    equals the Pearson correlation of ``s_it`` with ``y_t`` across units;
    ``aligned_it = gamma_it * y_t`` is the orthogonal projection of ``s_it``
    onto ``span{y_t}`` in the ``<.,.>_t`` geometry, and
    ``orthogonal_it = u_it = s_it - gamma_it * y_t`` satisfies
    ``<u_it, y_t>_t = 0`` and ``||u_it||_t^2 = 1 - gamma_it^2`` exactly, so
    ``s_it = aligned_it + u_it`` reconstructs the standardized forecast to
    machine precision. Aggregating: ``C_t = gamma_t gamma_t' + K_t`` with
    ``K_t = [<u_it,u_jt>_t]`` PSD (a Gram matrix).

    Fail-closed: raises ``ValueError`` on wrong shapes, non-finite entries,
    ``M < min_support``, no valid dates, or a perfectly aligned forecaster
    (``|gamma| = 1`` makes the Proposition 5 deviation correlation
    undefined); raises ``RuntimeError`` if any Proposition 1/5 identity is
    violated at runtime (which would indicate a numerical or coding fault).
    """
    s_raw = np.asarray(scores, dtype=float)
    y_raw = np.asarray(target, dtype=float)
    if s_raw.ndim != 3:
        raise ValueError("scores must be a T x N x M panel (dates, forecasters, units)")
    if y_raw.ndim != 2:
        raise ValueError("target must be a T x M matrix")
    t_raw, n, m = s_raw.shape
    if y_raw.shape != (t_raw, m):
        raise ValueError("target shape must match the scores panel's (T, M)")
    if not (np.all(np.isfinite(s_raw)) and np.all(np.isfinite(y_raw))):
        raise ValueError("scores and target must be finite; missing outputs are not imputed")
    if t_raw < 1 or n < 1:
        raise ValueError("panel must contain at least one date and one forecaster")
    min_support_i = _reject_bool_int(min_support, "min_support")
    if m < min_support_i:
        raise ValueError(f"common support must span >= {min_support_i} units (paper Appendix B)")
    if not (np.isfinite(eps) and eps > 0.0):
        raise ValueError("eps must be finite and positive")

    y_sd = y_raw.std(axis=1)
    s_sd = s_raw.std(axis=2)
    valid = (y_sd > eps) & np.all(s_sd > eps, axis=1)
    if not valid.any():
        raise ValueError("no valid dates: target or forecaster dispersion degenerate everywhere")
    s_v = s_raw[valid]
    y_v = y_raw[valid]
    s = _standardize_rows(s_v, 2)
    y = _standardize_rows(y_v, 1)

    # Definition 1: alignment is the unit-norm inner product = correlation.
    gamma = np.einsum("tnm,tm->tn", s, y) / m
    if np.abs(gamma).max() >= 1.0 - _GAMMA_PERFECT_TOL:
        raise ValueError(
            "a forecast is perfectly aligned with the target (|gamma| = 1); "
            "the Proposition 5 deviation correlation is undefined"
        )
    aligned = gamma[:, :, None] * y[:, None, :]
    orthogonal = s - aligned
    c_t = np.matmul(s, s.transpose(0, 2, 1)) / m
    k_t = np.matmul(orthogonal, orthogonal.transpose(0, 2, 1)) / m

    # Proposition 5: common-target deviation correlation.
    num = 1.0 + c_t - gamma[:, :, None] - gamma[:, None, :]
    den = 2.0 * np.sqrt((1.0 - gamma[:, :, None]) * (1.0 - gamma[:, None, :]))
    rho_e = num / den

    # Fail-closed runtime guards on the exact identities (Proposition 1).
    uy = np.einsum("tnm,tm->tn", orthogonal, y) / m
    recon = np.abs(aligned + orthogonal - s).max()
    gram_split = np.abs(k_t - (c_t - gamma[:, :, None] * gamma[:, None, :])).max()
    norm_ident = np.abs(
        np.einsum("tnm,tnm->tn", orthogonal, orthogonal) / m - (1.0 - gamma**2)
    ).max()
    try:
        min_eig = float(np.linalg.eigvalsh(k_t).min())
    except np.linalg.LinAlgError as exc:  # pragma: no cover - Gram matrices converge
        raise RuntimeError(f"orthogonal covariance eigendecomposition failed: {exc}") from exc
    if (
        np.abs(uy).max() > 1e-10
        or recon > 1e-10
        or gram_split > 1e-10
        or norm_ident > 1e-10
        or min_eig < -EPS_PSD
    ):
        raise RuntimeError(
            "Proposition 1 identity violated "
            f"(<u,y>={np.abs(uy).max():.3e}, recon={recon:.3e}, "
            f"gram={gram_split:.3e}, norm={norm_ident:.3e}, min_eig={min_eig:.3e})"
        )

    return AlignmentDecomposition(
        scores=np.ascontiguousarray(s),
        target=np.ascontiguousarray(y),
        gamma=np.ascontiguousarray(gamma),
        aligned=np.ascontiguousarray(aligned),
        orthogonal=np.ascontiguousarray(orthogonal),
        forecast_corr=np.ascontiguousarray(c_t),
        orthogonal_cov=np.ascontiguousarray(k_t),
        deviation_corr=np.ascontiguousarray(rho_e),
        valid_mask=np.ascontiguousarray(valid),
        n_excluded_dates=int(t_raw - valid.sum()),
    )


def composite_scores(decomp: AlignmentDecomposition, weights: Array) -> Array:
    """Combined standardized forecast ``s_w,t`` (any real weights, Prop. 2)."""
    w = _as_weights(decomp.n_forecasters, weights)
    return np.asarray(np.einsum("n,tnm->tm", w, decomp.scores), dtype=float)


def composite_losses(decomp: AlignmentDecomposition, weights: Array) -> Array:
    """Date-level relative-score losses ``||s_w,t - y_t||_t^2`` (T,)."""
    d = composite_scores(decomp, weights) - decomp.target
    return np.asarray(np.einsum("tm,tm->t", d, d) / decomp.n_units, dtype=float)


def relative_score_risk(decomp: AlignmentDecomposition, weights: Array) -> float:
    """Aggregated relative-score risk ``R(w) = sum_t a_t ||s_w,t - y_t||_t^2``.

    Uniform date weights ``a_t`` (paper Section 3); ``R(0) = 1`` is the
    no-information benchmark.
    """
    return float(composite_losses(decomp, weights).mean())


def _pooled_moments(decomp: AlignmentDecomposition, w: Array) -> tuple[float, float]:
    gbar = decomp.mean_gamma
    cbar = decomp.mean_forecast_corr
    g_w = float(w @ gbar)
    q_w = float(w @ cbar @ w)
    if not (np.isfinite(g_w) and np.isfinite(q_w)):
        raise ValueError("pooled moments are not finite")
    return g_w, q_w


def calibrated_composite_losses(decomp: AlignmentDecomposition, weights: Array) -> Array:
    """Date losses of the history-scale-calibrated composite ``||c_w s_w,t - y_t||_t^2``.

    ``c_w = g_w / q_w`` is the risk-minimizing scale (Prop. 3) estimated on
    the same dates, so the aggregate equals ``1 - rho_w^2`` exactly; plug-in
    scales affect aggregated risk only to second order (paper Appendix A).
    This is the loss basis of the scale-free three-way rule (Section 7: "each
    composite retaining its own history scale").
    """
    w = _as_weights(decomp.n_forecasters, weights)
    g_w, q_w = _pooled_moments(decomp, w)
    if q_w <= EPS_PSD:
        raise ValueError("composite has nonpositive squared norm; calibrated scale undefined")
    c_w = g_w / q_w
    d = c_w * composite_scores(decomp, w) - decomp.target
    return np.asarray(np.einsum("tm,tm->t", d, d) / decomp.n_units, dtype=float)


def risk_decomposition(decomp: AlignmentDecomposition, weights: Array) -> dict[str, float]:
    """Exact risk decompositions of Propositions 2, 3 and 6 plus Corollary 1.

    Returns a flat ``dict[str, float]`` of proper diagnostics; the identities
    ``R = 1 - 2 g_w + q_w`` (Prop. 2), ``R = (1 - rho_w^2) + q_w (1 - c_w)^2``
    (Prop. 3), ``R = (1 - g_w)^2 + w' K_pool w`` (Prop. 6) and
    ``R - 1 = -2 * margin`` (Cor. 1) are re-checked at runtime (fail-closed).
    """
    w = _as_weights(decomp.n_forecasters, weights)
    g_w, q_w = _pooled_moments(decomp, w)
    risk = relative_score_risk(decomp, w)
    if q_w <= EPS_PSD:
        raise ValueError("composite has nonpositive squared norm; scale statistics undefined")
    c_w = g_w / q_w
    rho2_w = g_w * g_w / q_w
    wt_gamma_t = decomp.gamma @ w  # (T,)
    align_within = float(np.mean((1.0 - wt_gamma_t) ** 2))
    k_bar = decomp.mean_orthogonal_cov
    orth_within = float(w @ k_bar @ w)
    align_pooled = (1.0 - g_w) ** 2
    orth_pooled = float(w @ decomp.pooled_orthogonal_cov @ w)
    out = {
        "risk": risk,
        "g_w": g_w,
        "q_w": q_w,
        "c_w": c_w,
        "rho2_w": rho2_w,
        "alignment_term_within": align_within,
        "orthogonal_term_within": orth_within,
        "alignment_term_pooled": align_pooled,
        "orthogonal_term_pooled": orth_pooled,
        "scale_free_risk": 1.0 - rho2_w,
        "scale_mismatch_penalty": q_w * (1.0 - c_w) ** 2,
        "margin_vs_no_information": g_w - 0.5 * q_w,  # Cor. 1: R < 1 iff > 0
    }
    checks = (
        abs(risk - (1.0 - 2.0 * g_w + q_w)),
        abs(risk - (align_within + orth_within)),
        abs(risk - (align_pooled + orth_pooled)),
        abs(risk - ((1.0 - rho2_w) + q_w * (1.0 - c_w) ** 2)),
        abs((risk - 1.0) + 2.0 * out["margin_vs_no_information"]),
    )
    if max(checks) > _IDENTITY_ATOL:
        raise RuntimeError(f"risk decomposition identity violated (max gap {max(checks):.3e})")
    return out


def attainable_risk(decomp: AlignmentDecomposition) -> dict[str, Any]:
    """Corollary 2: best in-sample risk of any linear combination.

    ``min_v {1 - 2 v' gamma_bar + v' C_bar v} = 1 - gamma_bar' C_bar^+
    gamma_bar``, attained at ``v* = C_bar^+ gamma_bar``; the quantity
    ``gamma_bar' C_bar^+ gamma_bar`` is the pooled squared multiple correlation
    of the standardized target on the forecasts. This is a bound for the
    sample moments, NOT an operationally attainable out-of-sample risk.
    """
    gbar = decomp.mean_gamma
    cbar = decomp.mean_forecast_corr
    v_star = np.asarray(np.linalg.pinv(cbar) @ gbar, dtype=float)
    value = float(gbar @ v_star)
    if not np.isfinite(value) or value < -_IDENTITY_ATOL or value > 1.0 + _IDENTITY_ATOL:
        raise ValueError(f"attainable-risk quadratic form outside [0, 1]: {value}")
    risk_at_v = relative_score_risk(decomp, v_star)
    if abs(risk_at_v - (1.0 - value)) > 1e-6:
        raise RuntimeError("Corollary 2 minimizer does not attain the stated risk")
    return {"attainable_risk": max(0.0, min(1.0, 1.0 - value)), "optimal_weights": v_star}


@dataclass(frozen=True)
class DilutionAccounting:
    """Propositions 8-9 accounting for adding ``batch`` to equal-weight ``pool``.

    Sign convention: ``delta < 0`` means admission *reduces* the pool's
    relative-score risk. ``delta_scale_free < 0`` is genuine improvement (the
    pooled squared correlation rises); ``delta_scale_mismatch`` is the dilution
    term (the composite's norm moves toward its risk-minimizing scale).
    ``pure_dilution`` flags the paper's failure mode: equal-weight admission
    fires (``delta < 0``) with no genuine improvement
    (``delta_scale_free >= 0``).
    """

    n_pool: int
    n_batch: int
    v_pool: float  # V_P
    v_batch: float  # V_A
    v_union: float  # V_{P u A}
    deviation_cov: float  # C^e_{PA}: pool-batch deviation covariance (Prop. 8)
    delta: float  # Delta_{A|P} = V_{P u A} - V_P (decision statistic)
    delta_scale_free: float  # Delta^sf = rho_P^2 - rho_{P u A}^2 (Prop. 9)
    delta_scale_mismatch: float  # Delta^scale (dilution term, Prop. 9)
    lambda_ew: float  # Lambda^EW_{k|P} (single candidate only; else NaN)
    rho2_pool: float
    rho2_union: float
    scale_pool: float  # c_P = g_P / q_P
    scale_union: float  # c_{P u A}
    scale_free_bound: float  # (m g_bar_A)^2 / (n^2 q_P); Prop. 9 side conditions
    admitted_by_equal_weight: bool  # delta < 0 (iff Lambda^EW < 1, Prop. 8)
    genuine_improvement: bool  # delta_scale_free < 0
    pure_dilution: bool  # admitted_by_equal_weight and not genuine_improvement


def dilution_accounting(
    decomp: AlignmentDecomposition,
    pool: Sequence[int] | NDArray[np.int64],
    batch: Sequence[int] | NDArray[np.int64],
) -> DilutionAccounting:
    """Decompose the incremental risk of equal-weight admission (Props. 8-9).

    ``pool`` (incumbent P, size n) and ``batch`` (candidates A, size m) are
    disjoint forecaster index sets; all composites are equal-weight over their
    members (``models.ensemble.equal_weights``). Proposition 8 (admission
    algebra): ``V_{P u A} = [n^2 V_P + m^2 V_A + 2 n m C^e_{PA}] / (n+m)^2``
    and, for a single candidate k,
    ``Delta_{k|P} = [V_k + 2 n C^e_{Pk} - (2n+1) V_P] / (n+1)^2`` with
    ``Lambda^EW_{k|P} = (V_k + 2 n C^e_{Pk}) / [(2n+1) V_P] < 1`` iff
    ``Delta_{k|P} < 0``. Proposition 9 (the dilution split):
    ``Delta_{A|P} = Delta^sf_{A|P} + Delta^scale_{A|P}`` with
    ``Delta^sf = rho_P^2 - rho_{P u A}^2`` and
    ``Delta^scale = q_{P u A} (1 - c_{P u A})^2 - q_P (1 - c_P)^2``, where
    ``rho^2_{P u A} = (n g_bar_P + m g_bar_A)^2 / (n^2 q_P + m^2 q_A +
    2 n m q_{PA})`` (Eq. (6)). Under weak alignment ``Delta ~= Delta^scale``:
    a candidate that merely shrinks the composite's norm reduces risk even
    when no alignment is detectable — the dilution that equal-weight
    admission mistakenly rewards. ``scale_free_bound`` is the Proposition 9
    bound ``-Delta^sf <= (m g_bar_A)^2 / (n^2 q_P)``, valid under the side
    conditions ``g_bar_P = 0`` and ``q_{PA} = 0`` (it shows the scale-free
    benefit of an aligned batch decays like ``(m/n)^2`` — equal-weight
    admission into a large pool is unresponsive to aligned candidates).

    Both Proposition 8 closed forms and the Proposition 9 split are exact
    identities and are re-checked at runtime (fail-closed ``RuntimeError``).
    """
    n_f = decomp.n_forecasters
    idx_p = _as_index_tuple(pool, n_f, "pool")
    idx_a = _as_index_tuple(batch, n_f, "batch")
    if set(idx_p) & set(idx_a):
        raise ValueError("pool and batch must be disjoint")
    n, m = len(idx_p), len(idx_a)

    w_p = np.zeros(n_f)
    w_p[list(idx_p)] = _equal_weights(n)
    w_a = np.zeros(n_f)
    w_a[list(idx_a)] = _equal_weights(m)
    w_u = np.zeros(n_f)
    w_u[list(idx_p)] = 1.0 / (n + m)
    w_u[list(idx_a)] = 1.0 / (n + m)

    v_p = float(composite_losses(decomp, w_p).mean())
    v_a = float(composite_losses(decomp, w_a).mean())
    v_u = float(composite_losses(decomp, w_u).mean())
    e_p = composite_scores(decomp, w_p) - decomp.target
    e_a = composite_scores(decomp, w_a) - decomp.target
    c_e = float(np.einsum("tm,tm->", e_p, e_a) / decomp.n_dates / decomp.n_units)

    closed_union = (n * n * v_p + m * m * v_a + 2.0 * n * m * c_e) / ((n + m) ** 2)
    delta = v_u - v_p
    if abs(closed_union - v_u) > _IDENTITY_ATOL:
        raise RuntimeError(f"Proposition 8 union identity violated ({closed_union} vs {v_u})")
    if v_p <= EPS_PSD:
        raise ValueError("pool risk V_P is degenerate (<= 0); admission algebra undefined")

    lam = float("nan")
    if m == 1:
        lam = (v_a + 2.0 * n * c_e) / ((2.0 * n + 1.0) * v_p)
        closed_single = (v_a + 2.0 * n * c_e - (2.0 * n + 1.0) * v_p) / ((n + 1.0) ** 2)
        if abs(closed_single - delta) > _IDENTITY_ATOL:
            raise RuntimeError(
                f"Proposition 8 single-candidate identity violated ({closed_single} vs {delta})"
            )
        if (lam < 1.0) != (delta < 0.0):
            raise RuntimeError("Proposition 8 Lambda^EW / Delta sign equivalence violated")

    g_p, q_p = _pooled_moments(decomp, w_p)
    g_u, q_u = _pooled_moments(decomp, w_u)
    if q_p <= EPS_PSD or q_u <= EPS_PSD:
        raise ValueError("composite squared norm degenerate; scale split undefined")
    c_p, c_u = g_p / q_p, g_u / q_u
    rho2_p, rho2_u = g_p * g_p / q_p, g_u * g_u / q_u
    delta_sf = rho2_p - rho2_u
    delta_scale = q_u * (1.0 - c_u) ** 2 - q_p * (1.0 - c_p) ** 2
    if abs(delta - (delta_sf + delta_scale)) > _IDENTITY_ATOL:
        raise RuntimeError("Proposition 9 scale-free/scale-mismatch split violated")

    # Eq. (6) cross-check: rho^2_{P u A} from pool/batch block moments.
    gbar = decomp.mean_gamma
    cbar = decomp.mean_forecast_corr
    g_bar_a = float(np.mean([gbar[j] for j in idx_a]))
    q_pa = float(w_p @ cbar @ w_a)
    num6 = (n * g_p + m * g_bar_a) ** 2
    den6 = n * n * q_p + m * m * float(w_a @ cbar @ w_a) + 2.0 * n * m * q_pa
    if den6 > EPS_PSD and abs(num6 / den6 - rho2_u) > 1e-7:
        raise RuntimeError("Proposition 9 Eq. (6) rho^2 identity violated")

    return DilutionAccounting(
        n_pool=n,
        n_batch=m,
        v_pool=v_p,
        v_batch=v_a,
        v_union=v_u,
        deviation_cov=c_e,
        delta=delta,
        delta_scale_free=delta_sf,
        delta_scale_mismatch=delta_scale,
        lambda_ew=lam,
        rho2_pool=rho2_p,
        rho2_union=rho2_u,
        scale_pool=c_p,
        scale_union=c_u,
        scale_free_bound=(m * g_bar_a) ** 2 / (n * n * q_p),
        admitted_by_equal_weight=bool(delta < 0.0),
        genuine_improvement=bool(delta_sf < 0.0),
        pure_dilution=bool(delta < 0.0 and delta_sf >= 0.0),
    )


@dataclass(frozen=True)
class SelectionStep:
    """One greedy step of the three-way rule (paper Section 7)."""

    pool_before: tuple[int, ...]
    candidates: tuple[int, ...]
    delta_hat: Array  # (J,) date-averaged loss differences
    se: Array  # (J,) Newey-West HAC standard errors (zeros if point_estimate)
    crit_value: float  # max-|t| critical value c (0.0 if point_estimate)
    decision: tuple[str, ...]  # per candidate: admit | reject | undecided
    min_detectable: Array  # (J,) delta + (c + z_{0.8}) * SE, reported per paper
    admitted: int | None


@dataclass(frozen=True)
class SelectionResult:
    """Result of greedy cautious selection (three-way rule, paper Section 7).

    ``guarantee`` restates Remark 1: per-step simultaneous bands cover all
    ``Delta_{A_j|P}`` with asymptotic probability ``1 - alpha`` conditional on
    the incumbent pool; the probability of admitting a candidate with
    ``Delta >= -delta`` is asymptotically at most ``alpha/2``, conditionally
    on ``P``. Greedy adaptation makes the path data-dependent: NO path-level
    error-rate control is claimed or provided.
    """

    selected: tuple[int, ...]
    start: int
    basis: str
    decision_rule: str
    alpha: float
    delta: float
    nw_lag: int
    n_draws: int
    seed: int
    steps: tuple[SelectionStep, ...]
    final_risk: float  # R of the equal-weight composite on ``selected``
    final_rho2: float  # pooled squared correlation of that composite
    guarantee: str


def _hac_covariance_of_means(d: Array, lag: int) -> Array:
    """Multivariate Bartlett HAC covariance of the date-mean of each column.

    Matrix generalization of ``metrics.inference.newey_west_variance`` (same
    kernel ``1 - h/(lag+1)``, same ``1/T`` normalization): ``Omega_{jl} =
    [Gamma_0 + sum_{h=1..L} w_h (Gamma_h + Gamma_h')] / T`` with
    ``Gamma_h = T^-1 sum_t u_jt u_l,t+h`` on centered columns.
    """
    t = d.shape[0]
    u = d - d.mean(axis=0)
    omega = (u.T @ u) / t
    for h in range(1, lag + 1):
        w = 1.0 - h / (lag + 1.0)
        g_h = (u[h:].T @ u[: t - h]) / t
        omega = omega + w * (g_h + g_h.T)
    omega = omega / t
    return np.asarray((omega + omega.T) / 2.0, dtype=float)


def _max_t_critical(
    omega: Array, se: Array, *, alpha: float, n_draws: int, rng: np.random.Generator, rel_tol: float
) -> float:
    """(1-alpha) quantile of max_j |Z_j| under the estimated HAC correlation.

    The covariance is normalized to a correlation matrix and its factor is
    truncated to eigenvalues exceeding ``rel_tol * lambda_max`` so
    non-positive-definite estimates are processed without repair (paper
    Section 7). Seeded draws keep the critical value deterministic.
    """
    j = se.size
    if j == 1:
        corr = np.ones((1, 1))
    else:
        corr = omega / np.outer(se, se)
        corr = np.clip(corr, -1.0, 1.0)
        np.fill_diagonal(corr, 1.0)
    try:
        eigvals, eigvecs = np.linalg.eigh((corr + corr.T) / 2.0)
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"HAC correlation eigendecomposition failed: {exc}") from exc
    lam_max = float(eigvals.max())
    if not np.isfinite(lam_max) or lam_max <= 0.0:
        raise ValueError("HAC correlation matrix has no positive eigenvalue")
    keep = eigvals > rel_tol * lam_max
    factor = eigvecs[:, keep] * np.sqrt(eigvals[keep])
    draws = factor @ rng.standard_normal((int(keep.sum()), n_draws))
    max_abs = np.max(np.abs(draws), axis=0)
    crit = float(np.quantile(max_abs, 1.0 - alpha))
    if not np.isfinite(crit) or crit <= 0.0:
        raise ValueError("max-|t| critical value is degenerate")
    return crit


def _candidate_loss_deltas(
    decomp: AlignmentDecomposition, pool: tuple[int, ...], cands: tuple[int, ...], basis: str
) -> Array:
    """Date-level differences ``d_{j,t} = l_{P u {j},t} - l_{P,t}`` (T, J).

    ``basis='equal_weight'``: raw equal-weight composites. ``basis=
    'scale_free'``: history-scale-calibrated composites (each composite keeps
    its own ``c_w = g_w / q_w`` from the same dates, Section 7), whose mean
    difference equals ``Delta^sf`` exactly in aggregate.
    """
    n_f = decomp.n_forecasters
    n = len(pool)
    w_p = np.zeros(n_f)
    w_p[list(pool)] = _equal_weights(n)
    j_arr = np.asarray(cands, dtype=int)
    s_pool = composite_scores(decomp, w_p)
    s_union = (n * s_pool[:, None, :] + decomp.scores[:, j_arr, :]) / (n + 1.0)
    if basis == "scale_free":
        g_p, q_p = _pooled_moments(decomp, w_p)
        if q_p <= EPS_PSD:
            raise ValueError("incumbent composite has degenerate squared norm")
        c_pool = g_p / q_p
        gbar = decomp.mean_gamma
        cbar = decomp.mean_forecast_corr
        g_u = (float(gbar[list(pool)].sum()) + gbar[j_arr]) / (n + 1.0)
        q_pa = cbar[np.ix_(list(pool), j_arr.tolist())].mean(axis=0)  # w_P' C_bar e_j
        q_u = (n * n * q_p + 2.0 * n * q_pa + 1.0) / ((n + 1.0) ** 2)
        if np.any(q_u <= EPS_PSD):
            raise ValueError("union composite has degenerate squared norm")
        c_union = g_u / q_u
        d_u = c_union[None, :, None] * s_union - decomp.target[:, None, :]
        d_p = c_pool * s_pool - decomp.target
    else:
        d_u = s_union - decomp.target[:, None, :]
        d_p = s_pool - decomp.target
    m_units = decomp.n_units
    ell_u = np.einsum("tjm,tjm->tj", d_u, d_u) / m_units
    ell_p = np.einsum("tm,tm->t", d_p, d_p) / m_units
    return np.asarray(ell_u - ell_p[:, None], dtype=float)


def cautious_selection(
    decomp: AlignmentDecomposition,
    *,
    basis: str = "equal_weight",
    decision_rule: str = "three_way",
    delta: float | None = None,
    alpha: float = 0.05,
    n_draws: int = 4000,
    seed: int = 20260922,
    nw_lag: int | None = None,
    eigen_rel_tol: float = 1e-8,
    max_steps: int | None = None,
) -> SelectionResult:
    """The paper's cautious three-way admission rule (Section 7).

    Greedy selection: start from the forecaster with the highest history
    alignment (highest squared alignment under the scale-free basis); at each
    step evaluate every remaining candidate ``A_j = {j}`` against the incumbent
    pool ``P`` via date-level loss differences ``d_{j,t}``; admit the candidate
    with the smallest ``Delta_hat_j`` among those cleared for admission;
    terminate once no candidate is admitted.

    Three-way decision with simultaneous HAC bands (``decision_rule=
    'three_way'``): Newey-West standard errors with lag
    ``floor(4 (T/100)^(2/9))`` (Newey-West 1994, per the paper); max-|t|
    critical value ``c`` = ``(1-alpha)`` quantile of ``max_j |Z_j|`` under the
    estimated HAC correlation (``n_draws`` seeded draws, eigenvalue-truncated
    factor). **Admit** iff ``Delta_hat_j + c*SE_j < -delta``; **reject** iff
    ``Delta_hat_j - c*SE_j > 0``; else **undecided**. The interval-inclusion
    logic follows equivalence testing (Schuirmann 1987); the minimum
    detectable effect ``delta + (c + z_{0.8}) SE_j`` is reported per step.

    ``decision_rule='point_estimate'`` is the paper's naive greedy-admission
    benchmark (admit iff ``Delta_hat_j < -delta``, no bands, no multiplicity
    adjustment): under weak alignment it can reward pure dilution, since
    ``Delta ~= Delta^scale`` for candidates that merely shrink the composite.

    ``basis='equal_weight'`` decides on raw equal-weight composite losses;
    ``basis='scale_free'`` decides on history-scale-calibrated composite
    losses, whose aggregate difference is exactly ``Delta^sf`` (genuine
    improvement only). Default margins follow the paper's Appendix B tuning
    grid: ``delta = 0.005`` (equal-weight) / ``0.0005`` (scale-free).

    Guarantee (Remark 1 — documented, not strengthened): conditional on the
    incumbent pool, the bands jointly cover all ``Delta_{A_j|P}`` with
    asymptotic probability ``1 - alpha``, so the probability of admitting a
    candidate with ``Delta_{A_j|P} >= -delta`` is asymptotically at most
    ``alpha/2``, *conditionally on P*. No path-level guarantee holds under
    greedy adaptation, and failing to clear admission is not evidence that
    predictive value is absent.
    """
    if basis not in ("equal_weight", "scale_free"):
        raise ValueError("basis must be 'equal_weight' or 'scale_free'")
    if decision_rule not in ("three_way", "point_estimate"):
        raise ValueError("decision_rule must be 'three_way' or 'point_estimate'")
    if not np.isfinite(alpha) or not (0.0 < alpha < 0.5):
        raise ValueError("alpha must lie in (0, 0.5)")
    if delta is None:
        margin = 0.005 if basis == "equal_weight" else 0.0005
    else:
        margin = float(delta)
        if not np.isfinite(margin) or margin < 0.0:
            raise ValueError("delta must be finite and nonnegative")
    n_draws_i = _reject_bool_int(n_draws, "n_draws")
    if n_draws_i < 100:
        raise ValueError("n_draws must be at least 100 for a stable max-|t| quantile")
    seed_i = _reject_bool_int(seed, "seed")
    if seed_i < 0:
        raise ValueError("seed must be nonnegative")
    if not np.isfinite(eigen_rel_tol) or not (0.0 < eigen_rel_tol < 1.0):
        raise ValueError("eigen_rel_tol must lie in (0, 1)")
    if max_steps is not None:
        max_steps_i = _reject_bool_int(max_steps, "max_steps")
        if max_steps_i < 0:
            raise ValueError("max_steps must be nonnegative")
    else:
        max_steps_i = decomp.n_forecasters - 1
    n_f = decomp.n_forecasters
    t = decomp.n_dates
    if nw_lag is None:
        lag = int(math.floor(4.0 * (t / 100.0) ** (2.0 / 9.0)))  # paper Section 7
    else:
        lag = _reject_bool_int(nw_lag, "nw_lag")
        if lag < 0 or lag >= max(t - 2, 1):
            raise ValueError("nw_lag must lie in [0, T-2)")
    lag = max(0, min(lag, max(t - 3, 0)))

    gbar = decomp.mean_gamma
    start = int(np.argmax(gbar if basis == "equal_weight" else gbar**2))
    rng = np.random.default_rng(seed_i)
    pool: list[int] = [start]
    steps: list[SelectionStep] = []
    guarantee = (
        "Per-step simultaneous HAC bands cover all Delta_{A_j|P} with asymptotic "
        f"probability {1.0 - alpha:.3f} conditional on the incumbent pool; the "
        "asymptotic probability of admitting a candidate with Delta >= -delta is "
        f"at most {alpha / 2.0:.4f}, conditionally on P (paper Remark 1). No "
        "path-level error control under greedy adaptation."
    )
    for _ in range(min(max_steps_i, n_f - 1)):
        cands = tuple(j for j in range(n_f) if j not in pool)
        if not cands:
            break
        d = _candidate_loss_deltas(decomp, tuple(pool), cands, basis)
        if not np.all(np.isfinite(d)):
            raise ValueError("candidate loss differences contain non-finite values")
        delta_hat = d.mean(axis=0)
        if decision_rule == "point_estimate":
            se = np.zeros(len(cands))
            crit = 0.0
        else:
            omega = _hac_covariance_of_means(d, lag)
            var_diag = np.diag(omega)
            if np.any(var_diag <= 0.0) or not np.all(np.isfinite(var_diag)):
                raise ValueError(
                    "degenerate loss-difference series (nonpositive HAC variance); "
                    "the three-way rule is undefined"
                )
            se = np.sqrt(var_diag)
            se_ref = np.asarray(
                [math.sqrt(max(newey_west_variance(d[:, j], lag), 0.0)) for j in range(d.shape[1])]
            )
            # Cross-check the multivariate diagonal against the composed
            # univariate estimator (same kernel and normalization).
            if np.max(np.abs(se - se_ref)) > 1e-10 + 1e-6 * np.max(se):
                raise RuntimeError("HAC diagonal disagrees with newey_west_variance")
            crit = _max_t_critical(
                omega,
                se,
                alpha=alpha,
                n_draws=n_draws_i,
                rng=rng,
                rel_tol=eigen_rel_tol,
            )
        admit = delta_hat + crit * se < -margin
        reject = delta_hat - crit * se > 0.0
        decision = tuple(
            "admit" if a else ("reject" if r else "undecided")
            for a, r in zip(admit, reject, strict=True)
        )
        min_det = margin + (crit + _Z_80) * se
        admitted: int | None = None
        if admit.any():
            admitted = cands[int(np.argmin(np.where(admit, delta_hat, np.inf)))]
        steps.append(
            SelectionStep(
                pool_before=tuple(pool),
                candidates=cands,
                delta_hat=np.ascontiguousarray(delta_hat),
                se=np.ascontiguousarray(se),
                crit_value=crit,
                decision=decision,
                min_detectable=np.ascontiguousarray(min_det),
                admitted=admitted,
            )
        )
        if admitted is None:
            break
        pool.append(admitted)

    w_sel = np.zeros(n_f)
    w_sel[pool] = _equal_weights(len(pool))
    g_sel, q_sel = _pooled_moments(decomp, w_sel)
    rho2_sel = g_sel * g_sel / q_sel if q_sel > EPS_PSD else 0.0
    return SelectionResult(
        selected=tuple(pool),
        start=start,
        basis=basis,
        decision_rule=decision_rule,
        alpha=float(alpha),
        delta=margin,
        nw_lag=lag,
        n_draws=n_draws_i if decision_rule == "three_way" else 0,
        seed=seed_i,
        steps=tuple(steps),
        final_risk=relative_score_risk(decomp, w_sel),
        final_rho2=float(rho2_sel),
        guarantee=guarantee,
    )


def error_correlation_mirror(decomp: AlignmentDecomposition) -> dict[str, float]:
    """Section 4/9.2 diagnostics: deviation correlation mirrors forecast corr.

    Proposition 5 makes ``rho^e`` an affine translation of ``rho^s`` at zero
    alignment (``rho^e = (1+rho^s)/2``), with departures driven by alignment
    alone — so error correlation is not a clean measure of forecast-output
    diversity. Returns off-diagonal diagnostics in two pooled flavors:
    date-level (every date/pair, includes within-date sampling noise) and
    pair-level (time-averaged ``C_bar`` / ``rho_bar^e`` per pair — the paper's
    Section 9.2 statistic, which reported R^2 = 0.9999 and a mean deviation
    correlation within 0.005 of the zero-alignment benchmark). Both report the
    zero-alignment benchmark, the translation gap, the Pearson correlation and
    affine-fit R^2 of ``rho^e`` on ``rho^s``, and the shares of negative
    entries (paper: forecast correlations can be negative while no deviation
    correlation is).
    """
    n = decomp.n_forecasters
    if n < 2:
        raise ValueError("error-correlation mirror needs at least two forecasters")
    off = ~np.eye(n, dtype=bool)
    rs = decomp.forecast_corr[:, off].reshape(-1)
    re = decomp.deviation_corr[:, off].reshape(-1)
    rs_pair = decomp.mean_forecast_corr[off]
    re_pair = decomp.mean_deviation_corr[off]
    for arr in (rs, re, rs_pair, re_pair):
        if not np.all(np.isfinite(arr)) or arr.size < 2:
            raise ValueError("deviation/forecast correlation diagnostics are degenerate")
    if float(np.std(rs)) <= 0.0 or float(np.std(rs_pair)) <= 0.0:
        raise ValueError("forecast correlations are constant; mirror diagnostics undefined")

    def _mirror(x: Array, z: Array) -> tuple[float, float]:
        pearson = float(np.corrcoef(x, z)[0, 1])
        slope, intercept = np.polyfit(x, z, 1)
        resid = z - (slope * x + intercept)
        ss_tot = float(np.sum((z - z.mean()) ** 2))
        r2 = 1.0 - float(resid @ resid) / ss_tot if ss_tot > 0.0 else float("nan")
        return pearson, r2

    pearson, r2 = _mirror(rs, re)
    pair_pearson, pair_r2 = _mirror(rs_pair, re_pair)
    benchmark = (1.0 + rs) / 2.0
    benchmark_pair = (1.0 + rs_pair) / 2.0
    return {
        "mean_forecast_corr": float(rs.mean()),
        "mean_deviation_corr": float(re.mean()),
        "zero_alignment_benchmark": float(benchmark.mean()),
        "translation_gap": float(re.mean() - benchmark.mean()),
        "mirror_pearson": pearson,
        "mirror_affine_r2": float(r2),
        "pair_mirror_pearson": pair_pearson,
        "pair_mirror_affine_r2": float(pair_r2),
        "pair_translation_gap": float(re_pair.mean() - benchmark_pair.mean()),
        "negative_forecast_corr_share": float(np.mean(rs < 0.0)),
        "negative_deviation_corr_share": float(np.mean(re < 0.0)),
        "min_deviation_corr": float(re.min()),
    }


def planted_panel(
    *,
    seed: int = 20260922,
    n_dates: int = 1600,
    n_units: int = 60,
    n_aligned: int = 6,
    n_diluters: int = 16,
    alignment: float = 0.06,
    cluster_corr: float = 0.30,
) -> tuple[Array, Array]:
    """Seeded SYNTHETIC validation world (correctness only, never market data).

    Mirrors the paper's simulation/positive-control construction (Sections
    8-9.5): one standardized common target; ``n_aligned`` forecasters with
    planted alignment ``s = a*y + sqrt(1-a^2)*iid noise`` (mutually correlated
    only through the target, ``rho^s ~= a^2``); ``n_diluters`` zero-alignment
    forecasters sharing one cluster factor, pairwise correlation
    ``cluster_corr``, each with maximal individual loss ``2(1-gamma) = 2``
    (high individual variance, no information). Returns RAW ``(scores,
    target)`` of shapes ``((T, k+m, M), (T, M))`` for ``alignment_decomposition``.
    Deterministic: identical outputs for identical arguments.
    """
    seed_i = _reject_bool_int(seed, "seed")
    if seed_i < 0:
        raise ValueError("seed must be nonnegative")
    t = _reject_bool_int(n_dates, "n_dates")
    m = _reject_bool_int(n_units, "n_units")
    k = _reject_bool_int(n_aligned, "n_aligned")
    q = _reject_bool_int(n_diluters, "n_diluters")
    if t < 30 or m < MIN_SUPPORT or k < 2 or q < 1:
        raise ValueError(
            f"planted world requires n_dates>=30, n_units>={MIN_SUPPORT}, "
            "n_aligned>=2, n_diluters>=1"
        )
    if not np.isfinite(alignment) or not (0.0 < alignment < 1.0):
        raise ValueError("alignment must lie in (0, 1)")
    if not np.isfinite(cluster_corr) or not (0.0 <= cluster_corr < 1.0):
        raise ValueError("cluster_corr must lie in [0, 1)")
    rng = np.random.default_rng(seed_i)
    y_raw = rng.normal(size=(t, m))
    y_std = _standardize_rows(y_raw, 1)
    a = float(alignment)
    aligned = a * y_std[:, None, :] + math.sqrt(1.0 - a * a) * rng.normal(size=(t, k, m))
    z = rng.normal(size=(t, m))  # shared cluster factor
    eps = rng.normal(size=(t, q, m))
    cc = float(cluster_corr)
    diluters = math.sqrt(cc) * z[:, None, :] + math.sqrt(1.0 - cc) * eps
    scores = np.concatenate([aligned, diluters], axis=1)
    return np.ascontiguousarray(scores), np.ascontiguousarray(y_raw)


def forecast_selection_benchmarks(*, seed: int = 20260922, fast: bool = False) -> dict[str, float]:
    """Seeded SYNTHETIC bench battery; flat ``dict[str, float]`` of diagnostics.

    World: 6 planted-aligned forecasters (alignment 0.06, target-mediated
    dependence only) + 16 zero-alignment diluters (cluster correlation 0.30)
    on 60 units — the paper's weak-alignment regime, where the full
    equal-weight pool sits above the no-information forecast. Reports: the
    Proposition 5 translation diagnostics (date-level and time-averaged pair
    level); per-diluter Proposition 9 accounting (the equal-weight admission
    failure mode: ``delta < 0`` and ``lambda_ew < 1`` with
    ``delta_scale_free > 0`` — admitted by scale mismatch alone); the
    three-way rule's pools on both bases, their dilution-loss removal
    fractions ``(R_full - R_sel) / (R_full - 1)``, and the point-estimate
    foil's diluter admissions. Proper diagnostics only (relative-score risks,
    IC-type correlations, admission statistics); no headline performance
    ratios; SYNTHETIC correctness, never market evidence. ``fast=True``
    shrinks the date count (800 vs 1600). Deterministic: bit-identical on
    repeated calls with the same arguments.
    """
    seed_i = _reject_bool_int(seed, "seed")
    scores, y = planted_panel(
        seed=seed_i,
        n_dates=800 if fast else 1600,
        n_units=60,
        n_aligned=6,
        n_diluters=16,
        alignment=0.06,
        cluster_corr=0.30,
    )
    decomp = alignment_decomposition(scores, y)
    k = 6
    aligned_pool = tuple(range(k))
    n_all = decomp.n_forecasters

    mirror = error_correlation_mirror(decomp)

    deltas: list[float] = []
    dsfs: list[float] = []
    dscales: list[float] = []
    lambdas: list[float] = []
    rewarded = 0
    for j in range(k, n_all):
        acc = dilution_accounting(decomp, aligned_pool, (j,))
        deltas.append(acc.delta)
        dsfs.append(acc.delta_scale_free)
        dscales.append(acc.delta_scale_mismatch)
        lambdas.append(acc.lambda_ew)
        rewarded += int(acc.pure_dilution and acc.admitted_by_equal_weight and acc.lambda_ew < 1.0)
    n_dil = n_all - k

    w_full = np.full(n_all, 1.0 / n_all)
    w_aligned = np.zeros(n_all)
    w_aligned[list(aligned_pool)] = _equal_weights(k)
    risk_full = relative_score_risk(decomp, w_full)
    risk_aligned = relative_score_risk(decomp, w_aligned)
    attainable = float(attainable_risk(decomp)["attainable_risk"])

    three_way = cautious_selection(decomp, seed=seed_i + 1)
    scale_free = cautious_selection(decomp, basis="scale_free", seed=seed_i + 1)
    naive = cautious_selection(decomp, decision_rule="point_estimate", seed=seed_i + 1)
    denom = risk_full - NO_INFORMATION_RISK
    if denom <= 0.0:
        raise RuntimeError("planted world failed to produce a dilution loss above no-information")
    out = {
        "seed": float(seed_i),
        "fast": float(bool(fast)),
        "n_dates": float(decomp.n_dates),
        "n_units": float(decomp.n_units),
        "n_aligned": float(k),
        "n_diluters": float(n_dil),
        "planted_alignment": 0.06,
        "planted_cluster_corr": 0.30,
        "mean_alignment": float(decomp.mean_gamma[:k].mean()),
        "diluter_mean_alignment": float(decomp.mean_gamma[k:].mean()),
        "mean_forecast_corr": mirror["mean_forecast_corr"],
        "mean_deviation_corr": mirror["mean_deviation_corr"],
        "zero_alignment_benchmark": mirror["zero_alignment_benchmark"],
        "deviation_translation_gap": mirror["translation_gap"],
        "deviation_mirror_pearson": mirror["mirror_pearson"],
        "deviation_mirror_affine_r2": mirror["mirror_affine_r2"],
        "pair_deviation_mirror_pearson": mirror["pair_mirror_pearson"],
        "pair_deviation_mirror_affine_r2": mirror["pair_mirror_affine_r2"],
        "pair_deviation_translation_gap": mirror["pair_translation_gap"],
        "negative_forecast_corr_share": mirror["negative_forecast_corr_share"],
        "negative_deviation_corr_share": mirror["negative_deviation_corr_share"],
        "diluter_delta_mean": float(np.mean(deltas)),
        "diluter_delta_scale_free_mean": float(np.mean(dsfs)),
        "diluter_delta_scale_mismatch_mean": float(np.mean(dscales)),
        "diluter_lambda_ew_mean": float(np.mean(lambdas)),
        "diluter_rewarded_share": float(rewarded) / float(n_dil),
        "no_information_risk": NO_INFORMATION_RISK,
        "equal_weight_full_risk": risk_full,
        "aligned_pool_risk": risk_aligned,
        "attainable_risk": attainable,
        "dilution_loss_full": denom,
        "three_way_ew_pool_size": float(len(three_way.selected)),
        "three_way_ew_diluters_admitted": float(sum(1 for j in three_way.selected if j >= k)),
        "three_way_ew_pool_risk": three_way.final_risk,
        "three_way_ew_pool_rho2": three_way.final_rho2,
        "three_way_ew_removal_fraction": (risk_full - three_way.final_risk) / denom,
        "three_way_sf_pool_size": float(len(scale_free.selected)),
        "three_way_sf_diluters_admitted": float(sum(1 for j in scale_free.selected if j >= k)),
        "three_way_sf_pool_risk": scale_free.final_risk,
        "three_way_sf_pool_rho2": scale_free.final_rho2,
        "three_way_sf_removal_fraction": (risk_full - scale_free.final_risk) / denom,
        "point_estimate_pool_size": float(len(naive.selected)),
        "point_estimate_diluters_admitted": float(sum(1 for j in naive.selected if j >= k)),
        "point_estimate_pool_risk": naive.final_risk,
        "point_estimate_pool_rho2": naive.final_rho2,
    }
    if not all(np.isfinite(v) for v in out.values()):
        raise RuntimeError("forecast-selection bench produced non-finite diagnostics")
    return out
