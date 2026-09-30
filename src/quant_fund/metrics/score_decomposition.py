"""Proper-score decompositions and the discrete/count proper-score canon.

Complements — never duplicates — the existing metrics suite:

- ``metrics/scoring.py`` (pinball, Gaussian / Student-t / mixture / empirical
  CRPS, QLIKE, Gaussian log score) is reused by import: ``crps_gaussian`` and
  ``mean_crps_gaussian`` back ``sharpness_decomposition``; ``crps_empirical``
  (plug-in spread estimator) is cross-checked against the fair (unbiased)
  ensemble CRPS implemented here.
- ``metrics/probability.py`` (binary Brier / log loss / ECE) stays the binary
  entry point; this module adds multi-category Brier, discrete logarithmic,
  spherical, and ranked probability scores for count / categorical outcomes.
- ``metrics/calibration2.py`` ``murphy_decomposition`` is the equal-width
  *binned* binary Murphy (1973) decomposition — its ``decomp_error`` carries
  the within-bin residual. ``brier_decomposition_binary`` here is the *exact*
  identity because it bins by distinct forecast values (sufficiency), and
  ``brier_decomposition`` generalizes it to multi-category outcomes.

Contents
--------
1. ``broecker_ensemble_crps_decomposition``: Bröcker (2012) three-term split
   of the expected ensemble CRPS into potential (intrinsic) uncertainty,
   quality, and reliability (extrinsic, forecast-attributable spread
   mismatch). Sample-exact when every term uses U-statistic (fair)
   estimators: ``crps = potential + quality - reliability``.
2. ``kolassa_crps_decomposition``: Kolassa (2016) discrete/count split
   ``mean CRPS = deviation + potential`` with forecast-free potential
   ``Σ_i Ĝ_i(1−Ĝ_i)``, generalized to per-observation forecasts through the
   exact per-threshold reliability/resolution split of Bröcker (2009):
   ``crps = reliability − resolution + potential``.
3. Discrete proper-score canon: ``ranked_probability_score`` (Epstein 1969),
   ``brier_multiclass`` (Brier 1950), ``log_score_discrete``,
   ``spherical_score`` (Good 1952) — all strictly proper; verified with
   seeded SYNTHETIC minimization-at-truth tests
   (``tests/unit/core/test_score_decomposition.py``).
4. ``sharpness_decomposition``: exact Gneiting-type split of mean Gaussian
   CRPS into a mean predictive-spread term and a mean error contribution.

References
----------
- Bröcker, J. (2009). Reliability, sufficiency, and the decomposition of
  proper scores. QJRMS 135(633), 1511–1519.
- Bröcker, J. (2012). Evaluating raw ensembles with the continuous ranked
  probability score. QJRMS 138(667), 1611–1617. doi:10.1002/qj.1891
- Kolassa, S. (2016). Evaluating predictive count data distributions in
  retail sales forecasting. International Journal of Forecasting 32(3),
  788–803.
- Epstein, E. S. (1969). A scoring system for probability forecasts of
  ranked categories. J. Appl. Meteor. 8(6), 985–987.
- Brier, G. W. (1950). Verification of forecasts expressed in terms of
  probability. Mon. Wea. Rev. 78(1), 1–3.
- Murphy, A. H. (1973). A new vector partition of the probability score.
  J. Appl. Meteor. 12(4), 1148–1153.
- Siegert, S. (2013). Variance estimation for Brier score decomposition.
  QJRMS 140(683), 1952–1959. (Traditional estimators; exactness requires
  binning by distinct forecast values.)
- Good, I. J. (1952). Rational decisions. JRSS-B 14(1), 107–114.
- Gneiting, T., Raftery, A. E. (2007). Strictly proper scoring rules,
  prediction, and estimation. JASA 102(477), 359–378.
- Zamo, M., Naveau, P. (2018). Estimation of the continuous ranked
  probability score with application to the analysis of the CRPS
  decomposition. Math. Geosci. 50, 277–306. (Fair ensemble CRPS.)

Research-diagnostic only — proper scores and their decompositions, never
headline Sharpe / P&L content, never live-trading or market evidence.
Elementwise scores mask invalid entries to honest NaN; decomposition
identities fail closed with ValueError on malformed input; empty means → NaN.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from quant_fund.metrics.scoring import crps_gaussian, mean_crps_gaussian
from quant_fund.utils.series import as_named_1d, require_same_length

__all__ = [
    "brier_decomposition",
    "brier_decomposition_binary",
    "brier_multiclass",
    "broecker_ensemble_crps_decomposition",
    "fair_ensemble_crps",
    "kolassa_crps_decomposition",
    "log_score_discrete",
    "mean_brier_multiclass",
    "mean_log_score_discrete",
    "mean_ranked_probability_score",
    "mean_spherical_score",
    "ranked_probability_score",
    "sharpness_decomposition",
    "spherical_score",
]

Array = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

# Renormalization tolerance mirroring scoring.crps_gaussian_mixture: a row sum
# within 1e-6 of 1 is a rounding artifact and is renormalized; anything larger
# is a caller bug and fails closed (NaN elementwise / ValueError in identities).
_PMF_SUM_TOL = 1e-6

# CRPS(N(mu, sigma^2), y = mu) / sigma = 2*phi(0) - 1/sqrt(pi).
_SHARPNESS_FACTOR = float(np.sqrt(2.0 / np.pi) - 1.0 / np.sqrt(np.pi))


def _as_1d(name: str, x: Array) -> Array:
    return as_named_1d(name, x)


def _require_same_length(*named: tuple[str, Array]) -> None:
    require_same_length(*named)


def _pmf_shape(pmfs: object, n_obs: int, *, name: str) -> Array:
    """Validate pmf matrix shape against ``n_obs``.

    A 1-d ``(K,)`` pmf is one fixed forecast for every observation (Kolassa's
    single-distribution evaluation setting) and is broadcast. A 2-d matrix
    must have exactly ``n_obs`` rows — a single-row 2-d matrix is *not*
    broadcast, so an accidental shape mismatch fails closed instead of
    silently scoring one forecast against many observations.
    """
    p = np.asarray(pmfs, dtype=float)
    if p.ndim == 1:
        p = np.repeat(p.reshape(1, -1), max(n_obs, 0), axis=0)
    if p.ndim != 2:
        raise ValueError(f"{name} must be a 2d (n, K) probability matrix")
    if p.shape[1] < 1:
        raise ValueError(f"{name} must have at least one category")
    if p.shape[0] != n_obs:
        raise ValueError(f"length mismatch: {name} rows={p.shape[0]}, y={n_obs}")
    return p


def _pmf_row_valid(p: Array) -> BoolArray:
    """Rows that are finite, nonnegative, and sum to 1 within tolerance."""
    sums = np.sum(p, axis=1)
    return (
        np.isfinite(p).all(axis=1)
        & np.all(p >= 0.0, axis=1)
        & (sums > 0.0)
        & (np.abs(sums - 1.0) < _PMF_SUM_TOL)
    )


def _pmf_lenient(pmfs: object, n_obs: int, *, name: str = "pmfs") -> tuple[Array, BoolArray]:
    """Shape-check, then renormalize valid rows; invalid rows score NaN."""
    p = _pmf_shape(pmfs, n_obs, name=name)
    valid = _pmf_row_valid(p)
    out = np.array(p, dtype=float, copy=True)
    sums = np.sum(out[valid], axis=1, keepdims=True)
    out[valid] = out[valid] / sums
    return out, valid


def _pmf_strict(pmfs: object, n_obs: int, *, name: str = "pmfs") -> Array:
    """Shape-check and fail closed unless every row is a valid pmf."""
    p = _pmf_shape(pmfs, n_obs, name=name)
    if p.shape[0] == 0:
        raise ValueError(f"{name} must be non-empty for a decomposition identity")
    finite = np.isfinite(p).all()
    if not finite:
        raise ValueError(f"{name} must be finite")
    if np.any(p < 0.0):
        raise ValueError(f"{name} must be nonnegative")
    sums = np.sum(p, axis=1)
    if np.any(np.abs(sums - 1.0) >= _PMF_SUM_TOL):
        raise ValueError(f"{name} rows must sum to 1 within {_PMF_SUM_TOL}")
    return p / sums[:, None]


def _support_grid(support: Array | None, k: int) -> Array:
    """Strictly increasing finite support grid aligned with pmf columns."""
    if support is None:
        return np.arange(k, dtype=float)
    s = np.asarray(support, dtype=float).reshape(-1)
    if s.size != k:
        raise ValueError(f"support must have one value per pmf category ({k})")
    if s.size == 0 or not np.isfinite(s).all():
        raise ValueError("support must be non-empty and finite")
    if s.size >= 2 and np.any(np.diff(s) <= 0.0):
        raise ValueError("support must be strictly increasing")
    return s


def _label_mask(y: Array, k: int) -> BoolArray:
    """Finite integer labels inside [0, K-1]."""
    mask = np.isfinite(y) & (y == np.floor(y)) & (y >= 0.0) & (y <= k - 1.0)
    return np.asarray(mask, dtype=bool)


def _onehot(labels: Array, k: int, mask: BoolArray) -> Array:
    """One-hot matrix; masked (invalid-label) rows stay all-zero."""
    n = labels.shape[0]
    out = np.zeros((n, k), dtype=float)
    idx = np.where(mask)[0]
    out[idx, labels[idx].astype(int)] = 1.0
    return out


def _mean_nonnan(scores: Array) -> float:
    """Mean over non-NaN entries, keeping +inf honest (log score at p=0).

    Unlike the ``mean_*`` wrappers in ``metrics/scoring.py`` (which drop
    non-finite values), a discrete log score of ``inf`` is real evidence that
    a realized outcome was predicted impossible — averaging must not hide it.
    Empty / all-NaN → honest NaN (research only).
    """
    values = scores[~np.isnan(scores)]
    if values.size == 0:
        return float("nan")
    return float(np.mean(values))


# ---------------------------------------------------------------------------
# Discrete / count proper-score canon
# ---------------------------------------------------------------------------


def ranked_probability_score(pmfs: Array, y: Array, support: Array | None = None) -> Array:
    r"""Elementwise ranked probability score (RPS). Epstein (1969).

    For a forecast pmf on the ordered support :math:`s_0 < \dots < s_{K-1}`
    with CDF :math:`F_i = \sum_{j \le i} p_j` and observation ``y``:

    .. math::

        \mathrm{RPS}(F, y) = \sum_{i=0}^{K-1}\bigl(F_i - 1\{y \le s_i\}\bigr)^2

    Epstein sums the first K−1 cumulative terms; the last term is identically
    zero for in-support observations (:math:`F_{K-1} = 1 = 1\{y \le s_{K-1}\}`),
    so summing over the full grid is equivalent. On the unit-spaced count grid
    (``support=None``, :math:`s_i = i`) this equals the discrete CRPS
    :math:`\sum_i (F_i - 1\{y \le i\})^2` used by Kolassa (2016) and equals
    ``crps_discrete`` in scoringRules; it is the ordinal analogue of
    ``brier_multiclass`` and strictly proper (verified SYNTHETIC).

    A 1-d ``(K,)`` pmf broadcasts as one fixed forecast for every observation.
    Invalid pmf rows (non-finite / negative / sum off 1 by ≥ 1e-6) or
    non-finite ``y`` → NaN at those indices (sum within 1e-6 is renormalized,
    mirroring ``scoring.crps_gaussian_mixture``). Length mismatch → ValueError.
    Empty → empty array. Research-diagnostic only — not a live claim.
    """
    y_arr = _as_1d("y", np.asarray(y, dtype=float))
    if y_arr.size == 0:
        return np.asarray([], dtype=float)
    p, valid = _pmf_lenient(pmfs, y_arr.size)
    grid = _support_grid(support, p.shape[1])
    out = np.full(y_arr.shape, np.nan, dtype=float)
    ok = valid & np.isfinite(y_arr)
    if not np.any(ok):
        return out
    cdf = np.cumsum(p[ok], axis=1)
    obs_cdf = (y_arr[ok, None] <= grid[None, :]).astype(float)
    out[ok] = np.sum((cdf - obs_cdf) ** 2, axis=1)
    return out


def mean_ranked_probability_score(pmfs: Array, y: Array, support: Array | None = None) -> float:
    """Mean RPS. Empty / all-NaN → honest NaN. Research-diagnostic only."""
    scores = ranked_probability_score(pmfs, y, support)
    if scores.size == 0:
        return float("nan")
    finite = scores[np.isfinite(scores)]
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite))


def brier_multiclass(pmfs: Array, y: Array) -> Array:
    r"""Elementwise multi-category Brier (quadratic) score.

    Brier (1950); Gneiting & Raftery (2007, §4.1). For category labels
    :math:`y_t \in \{0, \dots, K-1\}` and one-hot outcome :math:`o_t`:

    .. math::

        \mathrm{BS}(p_t, y_t) = \sum_{k=0}^{K-1} (p_{t,k} - o_{t,k})^2
        \in [0, 2]

    No 1/K normalization: with K=2 columns ``(1−p, p)`` this is exactly
    ``2 · (p − y)²``, i.e. twice ``metrics.probability.brier_score`` (pinned
    in tests). Strictly proper (verified SYNTHETIC). Non-integer or
    out-of-range labels, and invalid pmf rows, → NaN at those indices.
    Length mismatch → ValueError. Empty → empty array. Research-diagnostic
    only — never live Sharpe / promotion evidence.
    """
    y_arr = _as_1d("y", np.asarray(y, dtype=float))
    if y_arr.size == 0:
        return np.asarray([], dtype=float)
    p, valid = _pmf_lenient(pmfs, y_arr.size)
    k = p.shape[1]
    labels_ok = _label_mask(y_arr, k)
    ok = valid & labels_ok
    out = np.full(y_arr.shape, np.nan, dtype=float)
    if not np.any(ok):
        return out
    o = _onehot(y_arr, k, ok)
    out[ok] = np.sum((p[ok] - o[ok]) ** 2, axis=1)
    return out


def mean_brier_multiclass(pmfs: Array, y: Array) -> float:
    """Mean multi-category Brier. Empty / all-NaN → honest NaN. Research only."""
    scores = brier_multiclass(pmfs, y)
    if scores.size == 0:
        return float("nan")
    finite = scores[np.isfinite(scores)]
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite))


def log_score_discrete(pmfs: Array, y: Array) -> Array:
    r"""Elementwise discrete logarithmic score (loss orientation).

    Gneiting & Raftery (2007, eq. 22): :math:`S(p, y) = -\log p_y`. Strictly
    proper and local; its expectation at the truth is the Shannon entropy
    :math:`-\sum_k G_k \log G_k` (pinned in tests). A realized outcome with
    forecast probability 0 yields ``+inf`` — an honest, unbounded penalty; no
    epsilon padding is applied (unlike ``probability.log_loss`` for binary
    inputs, whose interior clip is documented there). Invalid pmf rows or
    non-integer / out-of-range labels → NaN. Length mismatch → ValueError.
    Empty → empty array. Research-diagnostic only.
    """
    y_arr = _as_1d("y", np.asarray(y, dtype=float))
    if y_arr.size == 0:
        return np.asarray([], dtype=float)
    p, valid = _pmf_lenient(pmfs, y_arr.size)
    k = p.shape[1]
    ok = valid & _label_mask(y_arr, k)
    out = np.full(y_arr.shape, np.nan, dtype=float)
    if not np.any(ok):
        return out
    idx = np.arange(int(y_arr.size))[ok]
    p_y = p[idx, y_arr[idx].astype(int)]
    with np.errstate(divide="ignore"):
        out[ok] = -np.log(p_y)
    return out


def mean_log_score_discrete(pmfs: Array, y: Array) -> float:
    """Mean discrete log score; ``inf`` stays ``inf`` (see ``_mean_nonnan``).

    Empty / all-NaN → honest NaN. Research-diagnostic only.
    """
    return _mean_nonnan(log_score_discrete(pmfs, y))


def spherical_score(pmfs: Array, y: Array) -> Array:
    r"""Elementwise spherical score (loss orientation). Good (1952).

    Gneiting & Raftery (2007, §4.1) report the reward form
    :math:`p_y / \lVert p \rVert_2`; this module returns the negatively
    oriented loss so every score here shares "lower is better":

    .. math::

        S(p, y) = 1 - \frac{p_y}{\lVert p \rVert_2} \in [0, 1)

    Strictly proper (verified SYNTHETIC); its expectation at the truth is
    :math:`1 - \lVert G \rVert_2`. Invalid pmf rows or non-integer /
    out-of-range labels → NaN. Length mismatch → ValueError. Empty → empty
    array. Research-diagnostic only.
    """
    y_arr = _as_1d("y", np.asarray(y, dtype=float))
    if y_arr.size == 0:
        return np.asarray([], dtype=float)
    p, valid = _pmf_lenient(pmfs, y_arr.size)
    k = p.shape[1]
    ok = valid & _label_mask(y_arr, k)
    out = np.full(y_arr.shape, np.nan, dtype=float)
    if not np.any(ok):
        return out
    idx = np.arange(int(y_arr.size))[ok]
    p_y = p[idx, y_arr[idx].astype(int)]
    norm = np.sqrt(np.sum(p[ok] ** 2, axis=1))
    out[ok] = 1.0 - p_y / norm
    return out


def mean_spherical_score(pmfs: Array, y: Array) -> float:
    """Mean spherical score. Empty / all-NaN → honest NaN. Research only."""
    scores = spherical_score(pmfs, y)
    if scores.size == 0:
        return float("nan")
    finite = scores[np.isfinite(scores)]
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite))


# ---------------------------------------------------------------------------
# Exact Brier decompositions (reliability − resolution + uncertainty)
# ---------------------------------------------------------------------------


def brier_decomposition(probs: Array, y: Array) -> dict[str, float]:
    r"""Exact multi-category Brier decomposition. Murphy (1973); Bröcker (2009).

    Bins are the *distinct forecast vectors* (sufficiency), so the identity

    .. math::

        \mathrm{BS} = \mathrm{REL} - \mathrm{RES} + \mathrm{UNC}

    holds exactly (to floating point), with

    - REL = :math:`\sum_b w_b \sum_k (\bar p_{b,k} - \bar o_{b,k})^2` —
      bin-mean forecast vs bin outcome frequency (miscalibration),
    - RES = :math:`\sum_b w_b \sum_k (\bar o_{b,k} - \bar o_k)^2` — outcome
      frequency refinement by forecast bin, the exact analogue of Murphy's
      estimator :math:`\pi_d = B_{\bullet d}/A_{\bullet d}` (Siegert 2013,
      eq. 18); equal-width binning of continuous forecasts leaves a
      within-bin residual (that residual is ``calibration2``'s
      ``decomp_error``),
    - UNC = :math:`\sum_k \bar o_k (1 - \bar o_k)` — Brier score of the
      outcome climatology (intrinsic uncertainty).

    Complements ``calibration2.murphy_decomposition`` (binary, equal-width
    bins, approximate): with K=2 columns ``(1−p, p)`` every term here is
    exactly twice the binary-scale term (pinned in tests).

    Fail-closed: non-finite / negative probabilities, row sums off 1 by
    ≥ 1e-6, non-integer or out-of-range labels, length mismatch, or empty
    input → ValueError. ``decomp_error`` is reported for auditability and is
    ~0 by construction. Research-diagnostic only.
    """
    y_arr = _as_1d("y", np.asarray(y, dtype=float))
    p = _pmf_strict(probs, y_arr.size)
    k = p.shape[1]
    if not np.all(_label_mask(y_arr, k)):
        raise ValueError("y must be integer labels in [0, K-1]")
    n = int(y_arr.size)
    o = _onehot(y_arr, k, np.ones(n, dtype=bool))
    _, inv = np.unique(p, axis=0, return_inverse=True)
    inv = np.asarray(inv).reshape(-1)
    n_bins = int(inv.max()) + 1 if n > 0 else 0
    counts = np.bincount(inv, minlength=n_bins).astype(float)
    p_bar = np.empty((n_bins, k), dtype=float)
    o_bar = np.empty((n_bins, k), dtype=float)
    for j in range(k):
        p_bar[:, j] = np.bincount(inv, weights=p[:, j], minlength=n_bins) / counts
        o_bar[:, j] = np.bincount(inv, weights=o[:, j], minlength=n_bins) / counts
    clim = o.mean(axis=0)
    w = counts / n
    rel = float(np.sum(w[:, None] * (p_bar - o_bar) ** 2))
    res = float(np.sum(w[:, None] * (o_bar - clim[None, :]) ** 2))
    unc = float(np.sum(clim * (1.0 - clim)))
    bs = float(np.mean(np.sum((p - o) ** 2, axis=1)))
    return {
        "brier": bs,
        "reliability": rel,
        "resolution": res,
        "uncertainty": unc,
        "decomp_error": bs - (rel - res + unc),
        "n_bins": float(n_bins),
        "n_obs": float(n),
    }


def brier_decomposition_binary(prob: Array, y: Array) -> dict[str, float]:
    r"""Exact binary Brier decomposition on the ``(p − y)²`` scale.

    Murphy (1973) / Siegert (2013) estimators with bins given by the distinct
    forecast values, so ``brier = reliability − resolution + uncertainty``
    holds exactly:

    - REL = :math:`\sum_b w_b (\bar p_b - \bar o_b)^2`,
    - RES = :math:`\sum_b w_b (\bar o_b - \bar o)^2` (outcome frequencies —
      Siegert 2013 eq. 18; under bin-wise calibration this equals the
      forecast-deviation form :math:`\sum_b w_b(\bar p_b - \bar o)^2`),
    - UNC = :math:`\bar o (1 - \bar o)`.

    When every distinct forecast value lands in its own equal-width bin, the
    terms match ``calibration2.murphy_decomposition`` exactly and that
    function's ``decomp_error`` vanishes (pinned in tests). ``brier`` equals
    ``probability.brier_score(prob, y)`` for valid inputs.

    Fail-closed: probabilities outside [0, 1] or non-finite, non-binary
    outcomes, length mismatch, or empty input → ValueError.
    Research-diagnostic only — not a live claim.
    """
    p = _as_1d("prob", np.asarray(prob, dtype=float))
    t = _as_1d("y", np.asarray(y, dtype=float))
    _require_same_length(("prob", p), ("y", t))
    if p.size == 0:
        raise ValueError("brier_decomposition_binary requires at least one observation")
    if not np.isfinite(p).all() or np.any(p < 0.0) or np.any(p > 1.0):
        raise ValueError("probabilities must be finite and lie in [0, 1]")
    if not np.all((t == 0.0) | (t == 1.0)):
        raise ValueError("outcomes must be binary")
    n = int(p.size)
    uniq, inv = np.unique(p, return_inverse=True)
    inv = np.asarray(inv).reshape(-1)
    counts = np.bincount(inv).astype(float)
    p_bar = np.bincount(inv, weights=p) / counts
    o_bar = np.bincount(inv, weights=t) / counts
    o_clim = float(np.mean(t))
    w = counts / n
    rel = float(np.sum(w * (p_bar - o_bar) ** 2))
    res = float(np.sum(w * (o_bar - o_clim) ** 2))
    unc = o_clim * (1.0 - o_clim)
    bs = float(np.mean((p - t) ** 2))
    return {
        "brier": bs,
        "reliability": rel,
        "resolution": res,
        "uncertainty": unc,
        "decomp_error": bs - (rel - res + unc),
        "n_bins": float(uniq.size),
        "n_obs": float(n),
    }


# ---------------------------------------------------------------------------
# Bröcker (2012): ensemble CRPS = potential + quality − reliability
# ---------------------------------------------------------------------------


def _mean_pairwise_distance_sorted(values: Array) -> float:
    r"""Unbiased E|X − X'| over distinct pairs (U-statistic), sorted form.

    For sorted :math:`x_{(0)} \le \dots \le x_{(n-1)}`:
    :math:`\sum_{i \ne j} |x_i - x_j| = 2 \sum_j (2j - n + 1)\, x_{(j)}`,
    so the unbiased mean pairwise distance is
    :math:`2 \sum_j (2j - n + 1) x_{(j)} / (n(n-1))`. Requires n ≥ 2.
    """
    xs = np.sort(np.asarray(values, dtype=float).reshape(-1))
    n = xs.size
    coeffs = 2.0 * np.arange(n, dtype=float) - float(n) + 1.0
    return float(2.0 * np.dot(coeffs, xs) / float(n * (n - 1)))


def _mean_pairwise_distance_rows(ensembles: Array) -> Array:
    """Row-wise unbiased mean pairwise distance for an (m, n) matrix, n ≥ 2."""
    xs = np.sort(np.asarray(ensembles, dtype=float), axis=1)
    n = xs.shape[1]
    coeffs = 2.0 * np.arange(n, dtype=float) - float(n) + 1.0
    return 2.0 * (xs @ coeffs) / float(n * (n - 1))


def fair_ensemble_crps(ensembles: Array, y: Array) -> Array:
    r"""Elementwise fair (unbiased) ensemble CRPS.

    Zamo & Naveau (2018); Efron (2011); the estimator underlying Bröcker
    (2012). For ensemble members :math:`x_1..x_n` and observation ``y``:

    .. math::

        \widehat{\mathrm{CRPS}}_{\mathrm{fair}}
        = \frac{1}{n}\sum_i |x_i - y|
        - \frac{1}{2n(n-1)}\sum_{i \ne j}|x_i - x_j|

    Unlike ``scoring.crps_empirical`` (plug-in, :math:`1/(2n^2)` spread sum),
    the spread term is an unbiased estimator of :math:`E|X - X'|`; the two
    relate exactly by
    ``crps_empirical = fair + U / (2n)`` with ``U`` the unbiased mean
    pairwise distance (pinned in tests). Rows with any non-finite member, or
    a non-finite ``y``, → NaN (honest mask). ``ensembles`` must be a finite-
    column 2-d ``(m, n)`` matrix with n ≥ 2 (ValueError otherwise — an
    unbiased spread needs two members). Empty → empty array.
    Research-diagnostic only.
    """
    e = np.asarray(ensembles, dtype=float)
    y_arr = _as_1d("y", np.asarray(y, dtype=float))
    if e.ndim != 2:
        raise ValueError("ensembles must be a 2d (m, n) matrix")
    n = int(e.shape[1])
    if n < 2:
        raise ValueError("fair ensemble CRPS needs at least 2 members per forecast")
    if e.shape[0] != y_arr.shape[0]:
        raise ValueError(f"length mismatch: ensembles rows={e.shape[0]}, y={y_arr.shape[0]}")
    if e.shape[0] == 0:
        return np.asarray([], dtype=float)
    out = np.full(y_arr.shape, np.nan, dtype=float)
    ok = np.isfinite(e).all(axis=1) & np.isfinite(y_arr)
    if not np.any(ok):
        return out
    e_ok = e[ok]
    mae = np.mean(np.abs(e_ok - y_arr[ok, None]), axis=1)
    spread = 0.5 * _mean_pairwise_distance_rows(e_ok)
    out[ok] = mae - spread
    return out


def broecker_ensemble_crps_decomposition(ensembles: Array, y: Array) -> dict[str, float]:
    r"""Bröcker (2012) three-term decomposition of the ensemble CRPS.

    Expected CRPS of raw ensembles :math:`X \sim F` against observations
    :math:`Y \sim \nu` splits into intrinsic and extrinsic parts:

    .. math::

        \mathrm{CRPS} = \mathrm{POT} + Q - R

    - **potential** :math:`\mathrm{POT} = \tfrac12 E|Y - Y'|` — intrinsic
      (irreducible) uncertainty of the observation process; the CRPS of a
      perfect forecast; depends on observations only,
    - **quality** :math:`Q = E|X - Y| - E|Y - Y'|` — extrinsic mismatch of
      ensemble members against observations; zero when members are
      distributionally identical to the truth,
    - **reliability** :math:`R = \tfrac12\bigl[E|X - X'| - E|Y - Y'|\bigr]`
      — ensemble-spread vs observation-variability mismatch; negative for
      underdispersed ensembles (penalizing via :math:`-R`), zero iff the
      mean pairwise member distance matches the mean pairwise observation
      distance.

    Sample version: ``ensembles`` is ``(m, n)``, ``y`` is ``(m,)``. Every
    expectation uses its unbiased U-statistic (mean pairwise distances over
    distinct pairs; cf. the fair CRPS of Zamo & Naveau 2018), which makes
    ``crps = potential + quality − reliability`` hold **exactly** on the
    sample — the observation-spread estimator cancels identically. ``crps``
    equals ``mean(fair_ensemble_crps(...))``.

    Fail-closed (ValueError): ``ensembles`` not 2-d / non-finite, m < 2 (no
    observation-pair U-statistic), n < 2 (no unbiased spread), ``y``
    non-finite or length-mismatched. Research-diagnostic only — never live
    Sharpe / promotion evidence; SYNTHETIC perfect-ensemble behavior is
    pinned in tests.
    """
    e = np.asarray(ensembles, dtype=float)
    y_arr = _as_1d("y", np.asarray(y, dtype=float))
    if e.ndim != 2:
        raise ValueError("ensembles must be a 2d (m, n) matrix")
    m, n = int(e.shape[0]), int(e.shape[1])
    if n < 2:
        raise ValueError("Bröcker decomposition needs at least 2 members per forecast")
    if m != y_arr.shape[0]:
        raise ValueError(f"length mismatch: ensembles rows={m}, y={y_arr.shape[0]}")
    if m < 2:
        raise ValueError("Bröcker decomposition needs at least 2 forecast/observation pairs")
    if not np.isfinite(e).all():
        raise ValueError("ensembles must be finite")
    if not np.isfinite(y_arr).all():
        raise ValueError("y must be finite")
    # A: member-vs-observation error term; S': mean unbiased ensemble spread.
    a_term = float(np.mean(np.abs(e - y_arr[:, None])))
    spread_mean = float(np.mean(_mean_pairwise_distance_rows(e)))
    # V: unbiased mean pairwise observation distance.
    v_term = _mean_pairwise_distance_sorted(y_arr)
    crps = a_term - 0.5 * spread_mean
    potential = 0.5 * v_term
    quality = a_term - v_term
    reliability = 0.5 * (spread_mean - v_term)
    return {
        "crps": crps,
        "potential": potential,
        "quality": quality,
        "reliability": reliability,
        "decomp_error": crps - (potential + quality - reliability),
        "mean_abs_error_term": a_term,
        "mean_spread_term": 0.5 * spread_mean,
        "observation_spread": v_term,
        "n_forecasts": float(m),
        "ensemble_size": float(n),
    }


# ---------------------------------------------------------------------------
# Kolassa (2016): discrete/count CRPS decomposition
# ---------------------------------------------------------------------------


def kolassa_crps_decomposition(
    pmfs: Array, y: Array, support: Array | None = None
) -> dict[str, float]:
    r"""Kolassa (2016) decomposition of the mean discrete (count) CRPS.

    With forecast CDF :math:`F_i` on the ordered support, observation
    indicator :math:`o_i = 1\{y \le s_i\}`, and the empirical outcome CDF
    :math:`\hat G_i = \frac1n \sum_t o_{t,i}`, the per-observation discrete
    CRPS (equal to ``ranked_probability_score`` on the unit grid) obeys the
    exact per-threshold binary-Brier identity (Murphy 1973 / Siegert 2013,
    binned by distinct forecast-CDF values), summed over thresholds:

    .. math::

        \overline{\mathrm{CRPS}}
        = \underbrace{\textstyle\sum_i \mathrm{REL}_i}_{\text{reliability}}
        - \underbrace{\textstyle\sum_i \mathrm{RES}_i}_{\text{resolution}}
        + \underbrace{\textstyle\sum_i \hat G_i (1 - \hat G_i)}_{\text{potential}}

    - **potential** — Kolassa (2016): forecast-free intrinsic uncertainty of
      the empirical outcome distribution (the discrete analogue of Bröcker's
      :math:`\tfrac12 E|Y - Y'|`),
    - **reliability** :math:`\mathrm{REL}_i = \sum_b w_{b} (\bar F_{b,i} -
      \bar o_{b,i})^2` — forecast-CDF vs conditional outcome frequency,
    - **resolution** :math:`\mathrm{RES}_i = \sum_b w_{b} (\bar o_{b,i} -
      \hat G_i)^2` — outcome refinement by forecast bin.

    For a single fixed forecast pmf (``pmfs`` 1-d, or all rows identical)
    every threshold has one bin, ``resolution == 0`` exactly, and
    ``reliability == deviation == Σ_i (F_i − Ĝ_i)²`` — Kolassa's two-term
    form ``mean CRPS = deviation + potential``. ``deviation`` is always
    reported (computed from the mean forecast CDF).

    Fail-closed (ValueError): empty input, non-finite ``y`` or pmfs,
    negative probabilities, row sums off 1 by ≥ 1e-6, length mismatch, or an
    invalid ``support`` grid. ``decomp_error`` (~0 by construction) and
    ``crps`` (== ``mean_ranked_probability_score`` on the same input) are
    reported for auditability. Research-diagnostic only — SYNTHETIC
    propriety is pinned in tests; never market evidence.
    """
    y_arr = _as_1d("y", np.asarray(y, dtype=float))
    p = _pmf_strict(pmfs, y_arr.size)
    if not np.isfinite(y_arr).all():
        raise ValueError("y must be finite")
    grid = _support_grid(support, p.shape[1])
    n, k = int(y_arr.size), int(p.shape[1])
    cdf = np.cumsum(p, axis=1)
    obs = (y_arr[:, None] <= grid[None, :]).astype(float)
    g_hat = obs.mean(axis=0)
    crps = float(np.mean(np.sum((cdf - obs) ** 2, axis=1)))
    potential = float(np.sum(g_hat * (1.0 - g_hat)))
    deviation = float(np.sum((cdf.mean(axis=0) - g_hat) ** 2))
    rel_total = 0.0
    res_total = 0.0
    for i in range(k):
        col = cdf[:, i]
        _, inv = np.unique(col, return_inverse=True)
        inv = np.asarray(inv).reshape(-1)
        counts = np.bincount(inv).astype(float)
        f_bar = np.bincount(inv, weights=col) / counts
        o_bar = np.bincount(inv, weights=obs[:, i]) / counts
        rel_total += float(np.sum(counts * (f_bar - o_bar) ** 2))
        res_total += float(np.sum(counts * (o_bar - g_hat[i]) ** 2))
    reliability = rel_total / n
    resolution = res_total / n
    return {
        "crps": crps,
        "potential": potential,
        "reliability": reliability,
        "resolution": resolution,
        "deviation": deviation,
        "decomp_error": crps - (reliability - resolution + potential),
        "n_obs": float(n),
        "n_support": float(k),
    }


# ---------------------------------------------------------------------------
# Gneiting-type sharpness decomposition for Gaussian CRPS
# ---------------------------------------------------------------------------


def sharpness_decomposition(y: Array, mu: Array, sigma: Array) -> dict[str, float]:
    r"""Exact split of mean Gaussian CRPS into spread and error contributions.

    Gneiting-type sharpness view of ``scoring.crps_gaussian`` (reused by
    import). With :math:`z = (y - \mu)/\sigma` and
    :math:`c = \sqrt{2/\pi} - 1/\sqrt{\pi} \approx 0.2337`:

    .. math::

        \mathrm{CRPS}
        = \underbrace{\sigma\, c}_{\text{sharpness}}
        + \underbrace{\sigma\bigl[z(2\Phi(z) - 1) + 2\varphi(z)
          - \sqrt{2/\pi}\bigr]}_{\text{error contribution} \ge 0}

    The sharpness term is the CRPS when the observation lands at the
    predictive median — the unavoidable cost of carrying predictive spread
    :math:`\sigma` (mean predictive spread contribution). The error term is
    zero at :math:`z = 0` and strictly positive otherwise (its derivative is
    :math:`2\sigma\,[\Phi(z) - 1/2]\,dz`-signed), i.e. the mean error
    contribution. On the valid-entry mask of ``crps_gaussian`` (finite
    ``y``/``mu``, :math:`\sigma > 0` finite):

    ``crps = sharpness + error_contribution`` exactly (``decomp_error`` ~ 0),
    and ``crps`` equals ``mean_crps_gaussian(y, mu, sigma)``.

    Empty / all-invalid → honest NaN dict (matching ``mean_crps_gaussian``).
    Length mismatch → ValueError. Research-diagnostic only — never live
    Sharpe / promotion evidence.
    """
    y_arr = _as_1d("y", y)
    mu_arr = _as_1d("mu", mu)
    sig_arr = _as_1d("sigma", sigma)
    _require_same_length(("y", y_arr), ("mu", mu_arr), ("sigma", sig_arr))
    nan_result = {
        "sharpness": float("nan"),
        "error_contribution": float("nan"),
        "crps": float("nan"),
        "decomp_error": float("nan"),
        "n": 0.0,
    }
    if y_arr.size == 0:
        return nan_result
    scores = crps_gaussian(y_arr, mu_arr, sig_arr)
    ok = np.isfinite(scores)
    if not np.any(ok):
        return nan_result
    sig_ok = sig_arr[ok]
    sharp_i = sig_ok * _SHARPNESS_FACTOR
    err_i = scores[ok] - sharp_i
    sharpness = float(np.mean(sharp_i))
    error_contribution = float(np.mean(err_i))
    crps = mean_crps_gaussian(y_arr, mu_arr, sig_arr)
    return {
        "sharpness": sharpness,
        "error_contribution": error_contribution,
        "crps": crps,
        "decomp_error": crps - (sharpness + error_contribution),
        "n": float(int(ok.sum())),
    }
