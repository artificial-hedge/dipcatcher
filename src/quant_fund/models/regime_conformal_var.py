"""Regime-weighted conformal VaR (wave-11 follow-up).

Conformal prediction-interval / VaR corrections whose calibration-set weights
depend on the current regime posterior. Given calibration nonconformity scores
``s_i`` observed in regimes described by hard labels or soft posteriors (rows
of an ``(n, K)`` matrix ``M``) and a current regime posterior ``pi`` over the
same ``K`` states, each calibration row receives weight

    w_i = sum_k M[i, k] * pi[k]                  (posterior weighting, default)
    w_i = sum_k M[i, k] * pi[k] / pi_cal[k]      (label-shift-corrected variant)

where ``pi_cal`` is the calibration regime marginal (column means of ``M``).
The VaR correction is the weighted conformal quantile of the scores under
``w`` (Tibshirani et al. 2019), so interval widths adapt to the current
regime while the marginal coverage level ``1 - alpha`` is preserved in the
weighted-exchangeability sense. With uniform posteriors the weights are
constant and the estimator reduces EXACTLY to unweighted split conformal
(closed-form check in the tests).

Regime posteriors may come from ``quant_fund.models.regime`` (online-filtered
``GaussianHMMRegime.predict_proba``), ``quant_fund.models.regime_switch``
(Hamilton filter), or any user-supplied K-state posterior; nothing in this
module is tied to a specific regime estimator.

References
----------
- Vovk, Gammerman & Shafer (2005), Algorithmic Learning in a Random World,
  Springer — algorithmic foundation of conformal prediction.
- Lei, G'Sell, Rinaldo, Tibshirani & Wasserman (2018), Distribution-free
  predictive inference for regression, JASA 113(523):1094-1111 — split
  conformal finite-sample marginal coverage.
- Tibshirani, Barber, Candes & Ramdas (2019), Conformal prediction under
  covariate shift, NeurIPS 2019, arXiv:1802.09184 — weighted split conformal;
  the intellectual ancestor here (regime posterior = shift covariate).
- Barber, Candes, Ramdas & Tibshirani (2023), The limits of distribution-free
  conditional predictive inference, Information and Inference 12(2):733-752,
  arXiv:2102.12401 — impossibility backdrop for the honesty note below.
- Hamilton (1989), A new approach to the economic analysis of nonstationary
  time series and the business cycle, Econometrica 57(2):357-384 — the regime
  machinery this module composes.

Honesty note (hard rule in this repo)
-------------------------------------
Regime-posterior weighting is a HEURISTIC localization device. The Tibshirani
et al. (2019) finite-sample marginal guarantee holds when the weights are the
exact likelihood ratios between the test-time and calibration-time covariate
(regime) distributions and the weighted scores are exchangeable. Per-regime
CONDITIONAL coverage has no finite-sample distribution-free guarantee
(Barber et al. 2023); with estimated HMM posteriors, filtering errors and
within-regime drift leak directly into coverage. Conditional coverage improves
over unweighted split conformal in the high-vol regime only insofar as the
posteriors are informative and regime-conditional score distributions are
stable between calibration and test. Every number produced by
``bench_regime_weighted_conformal_var`` is a SYNTHETIC oracle-regime
correctness check — never market evidence — and nothing here is a
live-trading claim. Lab scores are coverage and width. No Sharpe.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from quant_fund.metrics.conformal import (
    conformal_quantile,
    covered,
    cqr_scores,
    expand_interval,
    set_metrics,
)
from quant_fund.models.base import JoblibMixin, ModelMeta
from quant_fund.models.localized_conformal import effective_sample_size
from quant_fund.models.weighted_conformal import WEIGHT_CLIP, weighted_conformal_quantile

Array = NDArray[np.float64]

MIN_ESS: float = 12.0
# Floor on the calibration regime marginal in the label-shift ratio so an
# unseen regime cannot divide by zero; ratio weights are capped by weight_cap.
_MARGINAL_FLOOR: float = 1e-12

__all__ = [
    "MIN_ESS",
    "RegimeWeightedConformalVaR",
    "bench_regime_weighted_conformal_var",
    "onehot_regimes",
    "regime_coverage_report",
    "weighted_quantile_exact",
]


def _as_1d(x: Any) -> Array:
    return np.asarray(x, dtype=float).ravel()


def _hard_labels(arr: NDArray[Any], n_states: int | None) -> NDArray[np.int64]:
    """Validate a 1-d hard regime label vector; return int64 labels."""
    flat = np.asarray(arr).reshape(-1)
    if np.issubdtype(flat.dtype, np.integer):
        labels = flat.astype(np.int64)
    elif np.issubdtype(flat.dtype, np.floating):
        values = flat.astype(float)
        if not np.all(np.isfinite(values)) or not np.all(values == np.floor(values)):
            raise ValueError("hard regime labels must be integers in [0, n_states)")
        labels = values.astype(np.int64)
    else:
        raise ValueError("hard regime labels must be integers in [0, n_states)")
    if labels.size and int(labels.min()) < 0:
        raise ValueError("hard regime labels must be integers in [0, n_states)")
    if n_states is not None and labels.size and int(labels.max()) >= int(n_states):
        raise ValueError(f"regime label >= n_states ({int(n_states)})")
    return labels


def onehot_regimes(labels: NDArray[Any], n_states: int) -> Array:
    """One-hot encode validated hard regime labels into an (n, K) matrix."""
    if int(n_states) < 1:
        raise ValueError("n_states must be >= 1")
    lab = _hard_labels(np.asarray(labels), n_states)
    out = np.zeros((lab.size, int(n_states)), dtype=float)
    if lab.size:
        out[np.arange(lab.size), lab] = 1.0
    return out


def _posterior_matrix(mat: NDArray[Any], n_states: int | None) -> Array:
    """Validate and row-normalize a soft regime posterior matrix (n, K)."""
    m = np.asarray(mat, dtype=float)
    if m.ndim != 2 or m.shape[0] == 0 or m.shape[1] == 0:
        raise ValueError("regime posteriors must be a nonempty (n, K) matrix")
    if not np.all(np.isfinite(m)):
        raise ValueError("regime posteriors must be finite")
    if np.any(m < 0.0):
        raise ValueError("regime posteriors must be non-negative")
    if n_states is not None and m.shape[1] != int(n_states):
        raise ValueError(f"regime posteriors must have n_states={int(n_states)} columns")
    row_sums = m.sum(axis=1)
    if np.any(row_sums <= 0.0):
        raise ValueError("every regime posterior row must have positive mass")
    return np.asarray(m / row_sums[:, None], dtype=float)


def weighted_quantile_exact(values: Array, weights: Array, q: float) -> float:
    """Exact inverse-CDF weighted quantile (reusable; no interpolation).

    Returns the smallest sampled value ``v`` whose cumulative weight
    ``sum_{values <= v} w`` reaches ``q * sum(w)``. Ties are resolved toward
    the smaller value, and the result is always one of the sampled values.
    With uniform weights this coincides with
    ``numpy.quantile(..., method="inverted_cdf")``.

    Unlike :func:`quant_fund.models.weighted_conformal.weighted_conformal_quantile`,
    this routine carries no finite-sample ``+1 / (sum(w) + 1)`` point mass and
    therefore no conformal coverage correction — it is a descriptive weighted
    order statistic (e.g. a weighted empirical VaR level), not a prediction
    bound. Fail-closed: mismatched lengths, empty input, ``q`` outside [0, 1],
    non-finite values, negative or non-finite weights, and non-positive total
    weight raise ValueError.
    """
    v = _as_1d(values)
    w = _as_1d(weights)
    if v.size != w.size:
        raise ValueError("values and weights must have the same length")
    if v.size == 0:
        raise ValueError("values must be nonempty")
    if not np.isfinite(q) or not 0.0 <= float(q) <= 1.0:
        raise ValueError("q must lie in [0, 1]")
    if not np.all(np.isfinite(w)) or np.any(w < 0.0):
        raise ValueError("weights must be finite and non-negative")
    if not np.all(np.isfinite(v)):
        raise ValueError("values must be finite")
    total = float(np.sum(w))
    if total <= 0.0:
        raise ValueError("total weight must be positive")
    order = np.argsort(v, kind="mergesort")
    v_sorted = v[order]
    cumulative = np.cumsum(w[order]) / total
    idx = int(np.searchsorted(cumulative, float(q), side="left"))
    if idx >= v.size:
        return float(v_sorted[-1])
    return float(v_sorted[idx])


class RegimeWeightedConformalVaR(JoblibMixin):
    """Split-conformal VaR with regime-posterior calibration weights.

    Calibrate on nonconformity scores plus per-row regime information (hard
    labels or soft posteriors), then predict with the current regime posterior
    (from an online HMM filter or user-supplied). The correction is the
    Tibshirani et al. (2019) weighted conformal quantile under
    ``w_i = sum_k M[i, k] * pi[k]`` (see the module docstring for the
    label-shift-corrected variant and the honesty note: no finite-sample
    conditional guarantee unless the posteriors are exact).

    Degradation is explicit: when the current posterior puts all mass on
    regimes with no (or too little, per ``min_ess``) calibration support, the
    estimator falls back to the unweighted global conformal quantile rather
    than emitting an interval calibrated on effectively zero rows.
    """

    def __init__(
        self,
        alpha: float = 0.05,
        n_states: int | None = None,
        label_shift_correction: bool = False,
        min_ess: float | None = MIN_ESS,
        weight_cap: float = WEIGHT_CLIP[1],
    ) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        if n_states is not None and (
            isinstance(n_states, bool) or int(n_states) < 1 or not np.isfinite(float(n_states))
        ):
            raise ValueError("n_states must be an integer >= 1")
        if min_ess is not None and (not np.isfinite(float(min_ess)) or float(min_ess) <= 0.0):
            raise ValueError("min_ess must be None or > 0")
        if not np.isfinite(float(weight_cap)) or float(weight_cap) <= 0.0:
            raise ValueError("weight_cap must be finite and > 0")
        self.alpha = float(alpha)
        self.n_states = None if n_states is None else int(n_states)
        self.label_shift_correction = bool(label_shift_correction)
        self.min_ess = None if min_ess is None else float(min_ess)
        self.weight_cap = float(weight_cap)
        self.n_states_: int | None = None
        self.scores_: Array | None = None
        self.regime_matrix_: Array | None = None
        self.regime_marginal_: Array | None = None
        self.global_qhat: float = 0.0
        self.qhat: float | Array = 0.0

    # ------------------------------------------------------------------ fit

    def calibrate(self, scores: Array, regimes: NDArray[Any]) -> RegimeWeightedConformalVaR:
        """Store scores and the calibration regime matrix.

        ``regimes`` is either 1-d hard integer labels in ``[0, n_states)`` or
        a 2-d ``(n, K)`` soft posterior matrix (non-negative, positive row
        mass; rows are normalized). Fail-closed on any shape, finiteness,
        range, or alignment violation.
        """
        s = _as_1d(scores)
        if s.size == 0:
            raise ValueError("scores must be nonempty")
        if not np.all(np.isfinite(s)):
            raise ValueError("scores must be finite (NaN/inf rejected)")
        arr = np.asarray(regimes)
        if arr.ndim == 1:
            if arr.size != s.size:
                raise ValueError("regime labels must align with scores")
            matrix = onehot_regimes(_hard_labels(arr, self.n_states), self._infer_k(arr))
        elif arr.ndim == 2:
            if arr.shape[0] != s.size:
                raise ValueError("regime posteriors must align with scores")
            matrix = _posterior_matrix(arr, self.n_states)
        else:
            raise ValueError("regimes must be 1-d labels or a 2-d (n, K) posterior matrix")
        self.scores_ = s
        self.regime_matrix_ = matrix
        self.n_states_ = int(matrix.shape[1])
        self.regime_marginal_ = np.asarray(matrix.mean(axis=0), dtype=float)
        self.global_qhat = conformal_quantile(s, self.alpha)
        return self

    def _infer_k(self, arr: NDArray[Any]) -> int:
        if self.n_states is not None:
            return self.n_states
        labels = _hard_labels(arr, None)
        k = int(labels.max()) + 1 if labels.size else 0
        if k < 1:
            raise ValueError("cannot infer n_states from empty hard labels")
        return k

    # -------------------------------------------------------------- predict

    def _require_calibrated(self) -> None:
        if self.scores_ is None or self.regime_matrix_ is None or self.n_states_ is None:
            raise RuntimeError("RegimeWeightedConformalVaR must be calibrated before prediction")

    def _onehot(self, label: int) -> Array:
        k = int(self.n_states_ or 0)
        if not 0 <= int(label) < k:
            raise ValueError(f"regime label {int(label)} outside [0, {k})")
        vec = np.zeros(k, dtype=float)
        vec[int(label)] = 1.0
        return vec

    def _as_current_posterior(self, current_regime: Any) -> Array:
        """Coerce a scalar label, 0-d array, or length-K posterior to (K,)."""
        self._require_calibrated()
        k = int(self.n_states_ or 0)
        if isinstance(current_regime, (int, np.integer)) and not isinstance(current_regime, bool):
            return self._onehot(int(current_regime))
        arr = np.asarray(current_regime, dtype=float)
        if arr.ndim == 0 or (arr.ndim == 1 and arr.size == 1):
            value = float(arr.reshape(-1)[0])
            if not np.isfinite(value) or value != float(int(value)):
                raise ValueError("scalar current_regime must be an integer regime label")
            return self._onehot(int(value))
        if arr.ndim == 1 and arr.size == k:
            if not np.all(np.isfinite(arr)):
                raise ValueError("current regime posterior must be finite")
            if np.any(arr < 0.0):
                raise ValueError("current regime posterior must be non-negative")
            total = float(arr.sum())
            if total <= 0.0:
                raise ValueError("current regime posterior must have positive mass")
            return np.asarray(arr / total, dtype=float)
        raise ValueError(
            f"current_regime must be an integer label or a length-{k} posterior vector"
        )

    def weights(self, current_regime: Any) -> Array:
        """Calibration weights ``w_i`` for a current regime posterior (inspectable)."""
        posterior = self._as_current_posterior(current_regime)
        matrix = self.regime_matrix_
        marginal = self.regime_marginal_
        if matrix is None or marginal is None:  # pragma: no cover - guarded above
            raise RuntimeError("RegimeWeightedConformalVaR must be calibrated before prediction")
        if self.label_shift_correction:
            ratio = posterior / np.maximum(marginal, _MARGINAL_FLOOR)
            w = matrix @ ratio
        else:
            w = matrix @ posterior
        w = np.where(np.isfinite(w), w, 0.0)
        # Zero posterior mass must stay exactly zero (hard one-hot selection);
        # only the upper end is capped, unlike LocalizedCQR's floored clip.
        return np.asarray(np.clip(w, 0.0, self.weight_cap), dtype=float)

    def quantile_at(self, current_regime: Any) -> float:
        """Weighted conformal quantile qhat for one current regime posterior."""
        self._require_calibrated()
        w = self.weights(current_regime)
        assert self.scores_ is not None  # for mypy; calibrate() guarantees it
        if float(np.sum(w)) <= 0.0:
            return float(self.global_qhat)
        if self.min_ess is not None and effective_sample_size(w) < float(self.min_ess):
            return float(self.global_qhat)
        return weighted_conformal_quantile(self.scores_, w, self.alpha)

    def predict_rows(self, current_regimes: NDArray[Any]) -> Array:
        """Per-row qhats for a batch: 1-d integer labels or 2-d (m, K) posteriors."""
        self._require_calibrated()
        arr = np.asarray(current_regimes)
        if arr.ndim == 1:
            rows = onehot_regimes(_hard_labels(arr, self.n_states_), self.n_states_ or 1)
        elif arr.ndim == 2:
            rows = _posterior_matrix(arr, self.n_states_)
        else:
            raise ValueError("current_regimes must be 1-d labels or a 2-d (m, K) matrix")
        if rows.shape[0] == 0:
            return np.zeros(0, dtype=float)
        uniq, inv = np.unique(rows, axis=0, return_inverse=True)
        inv_flat = np.asarray(inv).reshape(-1)
        qhats = np.array([self.quantile_at(row) for row in uniq], dtype=float)
        return np.asarray(qhats[inv_flat], dtype=float)

    def predict(self, current_regime: Any) -> float | Array:
        """Dispatch on shape: scalar / single posterior -> float, batch -> Array.

        A 1-d array of length ``n_states`` is ALWAYS read as one soft
        posterior (float semantics); use :meth:`predict_rows` for batches of
        hard labels. Any other 1-d array is read as a batch of hard integer
        labels, and a 2-d ``(m, K)`` array as a batch of posteriors.
        """
        self._require_calibrated()
        n_states = self.n_states_
        assert n_states is not None  # for mypy; _require_calibrated() guarantees it
        arr = np.asarray(current_regime)
        if arr.ndim == 0:
            return self.quantile_at(arr)
        if arr.ndim == 1:
            if arr.size == 1:
                return self.quantile_at(arr[0])
            if arr.size == n_states:
                return self.quantile_at(arr)
            return self.predict_rows(arr)
        return self.predict_rows(arr)

    def predict_var(self, center: Array, current_regimes: Any, side: str = "lower") -> Array:
        """Conformal VaR bounds ``center -/+ qhat`` per row.

        ``current_regimes`` may be a single label/posterior broadcast to all
        rows, per-row 1-d integer labels, or a 2-d ``(m, K)`` posterior
        matrix. ``side='lower'`` is the long-position VaR bound (default);
        ``side='upper'`` is the short-side bound.
        """
        if side not in ("lower", "upper"):
            raise ValueError("side must be 'lower' or 'upper'")
        c = _as_1d(center)
        if c.size == 0:
            raise ValueError("center must be nonempty")
        if not np.all(np.isfinite(c)):
            raise ValueError("center must be finite")
        self._require_calibrated()
        n_states = self.n_states_
        assert n_states is not None  # for mypy; _require_calibrated() guarantees it
        k = int(n_states)
        arr = np.asarray(current_regimes)
        if arr.ndim == 0 or (
            arr.ndim == 1 and (arr.size == 1 or (arr.size == k and arr.size != c.size))
        ):
            q = np.full(c.size, self.quantile_at(arr))
        elif arr.ndim == 1 and arr.size == c.size and arr.size != k:
            q = self.predict_rows(arr)
        elif arr.ndim == 2:
            if arr.shape[0] == 1 and c.size > 1:
                q = np.full(c.size, self.quantile_at(arr[0]))
            elif arr.shape[0] == c.size:
                q = self.predict_rows(arr)
            else:
                raise ValueError("current_regimes must align with center or be a single row")
        else:
            raise ValueError(
                "current_regimes is ambiguous or misaligned: pass a single label/posterior, "
                "per-row labels of length len(center), or a (len(center), n_states) matrix"
            )
        qhat = np.asarray(q, dtype=float)
        self.qhat = qhat
        return c - qhat if side == "lower" else c + qhat

    # ------------------------------------------------------------- metadata

    def metadata(self) -> ModelMeta:
        extra: dict[str, Any] = {
            "alpha": self.alpha,
            "n_states": self.n_states_,
            "label_shift_correction": self.label_shift_correction,
            "min_ess": self.min_ess,
            "weight_cap": self.weight_cap,
            "global_qhat": self.global_qhat,
        }
        return ModelMeta(
            family="conformal",
            name="regime_weighted_conformal_var",
            version="v1",
            extra=extra,
        )


def regime_coverage_report(
    y: Array,
    lower: Array,
    upper: Array,
    regimes: NDArray[Any],
) -> dict[str, Any]:
    """Unconditional AND per-regime conditional coverage of prediction intervals.

    ``regimes`` is 1-d hard labels or a 2-d posterior matrix (stratified by
    argmax). Every regime group is reported (no minimum-count skip) together
    with its observation count, so callers can judge thin strata themselves.
    Fail-closed on empty input, length mismatch, non-integer labels, or
    invalid posteriors. Coverage is a proper lab diagnostic here; never a
    market-evidence claim.
    """
    y_a = _as_1d(y)
    lo_a = _as_1d(lower)
    hi_a = _as_1d(upper)
    if y_a.size == 0 or y_a.size != lo_a.size or y_a.size != hi_a.size:
        raise ValueError("y, lower, and upper must be nonempty and of equal length")
    arr = np.asarray(regimes)
    if arr.ndim == 1:
        if arr.size != y_a.size:
            raise ValueError("regime labels must align with y")
        labels = _hard_labels(arr, None)
    elif arr.ndim == 2:
        if arr.shape[0] != y_a.size:
            raise ValueError("regime posteriors must align with y")
        matrix = _posterior_matrix(arr, None)
        labels = np.asarray(np.argmax(matrix, axis=1), dtype=np.int64)
    else:
        raise ValueError("regimes must be 1-d labels or a 2-d (n, K) posterior matrix")
    hits = covered(y_a, lo_a, hi_a)
    width = hi_a - lo_a
    finite = np.isfinite(y_a) & np.isfinite(lo_a) & np.isfinite(hi_a)
    if not bool(finite.any()):
        raise ValueError("y, lower, and upper must contain at least one finite row")
    hits, width, labels = hits[finite], width[finite], labels[finite]
    per_cov: dict[str, float] = {}
    per_width: dict[str, float] = {}
    per_n: dict[str, float] = {}
    for g in np.unique(labels):
        sel = labels == g
        key = str(int(g))
        per_cov[key] = float(np.mean(hits[sel]))
        per_width[key] = float(np.mean(width[sel]))
        per_n[key] = float(int(sel.sum()))
    return {
        "unconditional_coverage": float(np.mean(hits)),
        "mean_width": float(np.mean(width)),
        "n": float(int(hits.size)),
        "per_regime_coverage": per_cov,
        "per_regime_mean_width": per_width,
        "per_regime_n": per_n,
    }


# Bench DGP: two-regime Gaussian with different vols, drawn from the repo's
# own regime simulator (quant_fund.stress.regimes). State 1 is the high-vol
# regime; the stationary high-vol share is 0.03 / (0.03 + 0.05) = 0.375.
_BENCH_BAND: float = 0.25


def _synthetic_regime_stream(
    n_cal: int, n_test: int, seed: int
) -> tuple[Array, NDArray[np.int64], Array, NDArray[np.int64]]:
    """SYNTHETIC two-regime Gaussian stream with oracle state labels."""
    # Lazy: the regime simulator lives in the research layer — a deferred
    # import is the sanctioned layer-order cycle-breaker
    # (docs/ARCHITECTURE_GUARDS.md); only the SYNTHETIC bench DGP needs it.
    from quant_fund.stress.regimes import GaussianRegimeSpec, simulate_gaussian_hmm

    spec = GaussianRegimeSpec(
        means=np.array([[0.0], [0.0]]),
        covs=np.array([[[0.5**2]], [[2.0**2]]]),
        transition=np.array([[0.97, 0.03], [0.05, 0.95]]),
    )
    rng = np.random.default_rng(seed)
    returns, states = simulate_gaussian_hmm(spec, int(n_cal) + int(n_test), rng)
    y = np.asarray(returns[:, 0], dtype=float)
    return y[:n_cal], states[:n_cal], y[n_cal:], states[n_cal:]


def bench_regime_weighted_conformal_var(
    n_cal: int = 600,
    n_test: int = 400,
    alpha: float = 0.10,
    seed: int = 11,
    label_shift_correction: bool = False,
) -> dict[str, float | str]:
    """Coverage on a SYNTHETIC two-regime HMM stream with ORACLE posteriors.

    Correctness check only: the current regime posterior at test time is the
    one-hot of the simulator's true state, so this measures what regime
    weighting achieves with perfect regime information. It is labeled
    ``dgp=fixture`` / ``claim=research_metric_only`` and is never market
    evidence. Lab scores are coverage and width. No Sharpe.
    """
    if isinstance(n_cal, bool) or int(n_cal) < 1 or isinstance(n_test, bool) or int(n_test) < 1:
        raise ValueError("n_cal and n_test must be positive integers")
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    y_c, s_c, y_t, s_t = _synthetic_regime_stream(int(n_cal), int(n_test), int(seed))
    lo_c = np.full(y_c.size, -_BENCH_BAND)
    hi_c = np.full(y_c.size, _BENCH_BAND)
    lo_t = np.full(y_t.size, -_BENCH_BAND)
    hi_t = np.full(y_t.size, _BENCH_BAND)
    scores = cqr_scores(y_c, lo_c, hi_c)
    model = RegimeWeightedConformalVaR(
        alpha=float(alpha), n_states=2, label_shift_correction=label_shift_correction
    ).calibrate(scores, s_c)
    qhats = model.predict_rows(onehot_regimes(s_t, 2))
    wlo, whi = expand_interval(lo_t, hi_t, qhats)
    q_u = conformal_quantile(scores, float(alpha))
    ulo, uhi = expand_interval(lo_t, hi_t, q_u)
    weighted = set_metrics(y_t, wlo, whi)
    plain = set_metrics(y_t, ulo, uhi)
    w_report = regime_coverage_report(y_t, wlo, whi, s_t)
    u_report = regime_coverage_report(y_t, ulo, uhi, s_t)
    w_high = float(w_report["per_regime_coverage"].get("1", float("nan")))
    u_high = float(u_report["per_regime_coverage"].get("1", float("nan")))
    return {
        "synthetic_coverage": weighted.coverage,
        "synthetic_mean_width": weighted.mean_width,
        "synthetic_median_width": weighted.median_width,
        "synthetic_highvol_coverage": w_high,
        "synthetic_lowvol_coverage": float(w_report["per_regime_coverage"].get("0", float("nan"))),
        "synthetic_highvol_mean_width": float(
            w_report["per_regime_mean_width"].get("1", float("nan"))
        ),
        "synthetic_lowvol_mean_width": float(
            w_report["per_regime_mean_width"].get("0", float("nan"))
        ),
        "synthetic_unweighted_coverage": plain.coverage,
        "synthetic_unweighted_mean_width": plain.mean_width,
        "synthetic_unweighted_highvol_coverage": u_high,
        "synthetic_unweighted_lowvol_coverage": float(
            u_report["per_regime_coverage"].get("0", float("nan"))
        ),
        "synthetic_highvol_coverage_gain": w_high - u_high,
        "synthetic_n": float(weighted.n),
        "synthetic_alpha": float(alpha),
        "synthetic_seed": float(seed),
        "synthetic_dgp": "fixture",
        "synthetic_claim": "research_metric_only",
        "synthetic_regimes": "oracle_true_states_synthetic",
    }
