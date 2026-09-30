"""Prediction-Interval-Conditional Prediction Intervals (PICPIs).

Yang, Huang, Hou, Imbens & Jordan (2026), "PICPIs: Prediction-Interval-
Conditional Prediction Intervals", arXiv:2609.25388 [stat.ML] (45 pp).

A PICPI is an interval I in *prediction-value space* satisfying the
self-consistency condition (their Definition 2.1, Eq. (5), binary outcomes)

    P(Y = 1 | p(X) in I) in I,

i.e. the interval simultaneously defines a stratum of prediction values and
certifies that the stratum-conditional outcome probability lies in the same
interval. Remark 3.6 extends the definition to regression: for a predictor of
E[Y | X] and bounded Y in [y_lo, y_hi] the condition reads

    E[Y | p(X) in I] in I,

and the Section-2 constructions carry over by affine rescaling of Y and p(X)
to [0, 1] (implemented here via ``outcome_range``). A PICPI is NOT a
prediction interval for the outcome Y of a single observation and carries no
1 - alpha marginal-coverage claim for Y; Theorem 4.1 / Eq. (13) instead bound
miscoverage of the *true probability* p*(X) by the epsilon-expanded interval.

What is implemented (algorithm names refer to the fetched paper text):

1. ``fit_picpi`` — Algorithm 1 (population mode). Grid {0, 1/K, ..., 1} of
   the rescaled prediction range; every candidate [a, b] = [i/K, j/K] with
   i < j is admitted when it holds N > 0 calibration predictions and the
   empirical outcome mean in [a, b] lies in the *margin-shrunk* interval
   (their Line 6)

       [a + 2 sqrt(log(K^2/delta)/N),  b - 2 sqrt(log(K^2/delta)/N)].

   Guarantee (finite-sample, Theorem 2.2 + Appendix A.1): if the calibration
   pairs are i.i.d., Y is bounded in ``outcome_range``, and p is fixed
   independently of the calibration sample, then with probability >= 1 -
   delta EVERY admitted interval is a genuine population PICPI. The proof is
   Hoeffding plus a union bound over the K^2 grid candidates — there is no
   exchangeability/conformal argument anywhere in the paper; the guarantee
   mechanism is a concentration-margin screen, not a conformal wrapper.
   Note the paper's margin constant is deliberately conservative: plugging
   2 sqrt(log(K^2/delta)/N) into Hoeffding gives a per-candidate failure
   bound 2 (delta/K^2)^8, far below the sqrt(log(2K^2/delta)/(2N)) that a
   tight two-sided Hoeffding union bound would need. The optional
   ``relax_endpoints=True`` implements their Eq. (21): the margin is skipped
   at the hard boundaries (a = y_lo, b = y_hi) where the conditional mean
   cannot exit [0, 1] anyway.
2. ``fit_picpi_empirical`` — Algorithm 3 (empirical mode, their Section
   5.4): sort predictions; greedily grow a partition left-to-right, choosing
   the smallest grid endpoint t_j such that the empirical outcome mean over
   (t_{j-1}, t_j] lies in (t_{j-1}, t_j] AND the empirical mean over the
   remainder (t_j, 1] lies in (t_j, 1] (the remainder condition is what
   makes the recursion well defined; it is waived at t_j = 1). Honest
   scope: Algorithm 3 omits the concentration margin, so it certifies
   self-consistency only with respect to the *empirical* calibration
   measure; the paper states explicitly that the population validity of
   Theorem 2.2 no longer applies. Fail-closed edge choices (unspecified in
   the pseudocode, documented below): bins with zero calibration count are
   never admitted (the empirical mean is 0/0, so Line 5 is not "subject
   to" satisfaction); if no endpoint qualifies — including t_j = 1, where
   only the bin condition is checked — the remaining tail is left
   UNCERTIFIED and queries there return the trivial fallback.
3. Query interface — Algorithm 1's Inference procedure returns the SHORTEST
   admitted interval containing the new prediction value (ties broken by
   the smaller lower endpoint, deterministic); when no interval contains it,
   Section 4.1 documents the fallback [l(z), u(z)] = [y_lo, y_hi], the
   trivial interval, which is always a population PICPI because a
   conditional mean of a ``outcome_range``-bounded outcome lies in the
   range. ``PicpiFit.expand`` implements Eq. (13): the epsilon-expansion
   C_tp(x; eps) = E^eps([l(p(x)), u(p(x))]) clipped to the outcome range.
   ``PicpiFit.coverage_profile`` evaluates Eq. (6): the fraction of
   prediction values enclosed by some admitted interval of width <= a cap,
   which Theorem 3.2 lower-bounds by 1 - delta_1 at cap
   C(n, eps, delta, delta_1) = c0 log(1/delta_1) ((log(n/delta)/(lambda n))^{1/3}
   + eps/delta_1) under lambda-regularity of p(X) (Definition 3.1) and L1
   prediction error eps — widths shrink at n^{-1/3} up to log factors, but
   on a FIXED K-grid they floor at 1/K, so demonstrating the rate requires
   the grid to be finer than the theoretical width.
4. ``disjoint_calibration`` — Algorithm 4 (Appendix C): length filter plus
   maximum-covered-length weighted interval scheduling (sort by right
   endpoint, binary-search compatibility, DP, backtrack) yielding a
   disjoint sub-collection.
5. Multiclass label sets — Section 4.2: per-class one-vs-rest Algorithm 1
   with failure budget delta/G (Theorem 4.5 requires classwise budgets
   summing to delta_PI), disjointified via Algorithm 4, then the empirical
   relaxation of Algorithm 2: masses m_hat and priors pi_hat from the SAME
   calibration sample, Hoeffding radii eps_m = sqrt(log(2 N_cand/delta_m)/
   (2 n)) and eps_pi = sqrt(log(2 G/delta_pi)/(2 n)) with N_cand the total
   number of grid candidates searched, and per class the keep-side knapsack
   (min sum m_hat s.t. sum a_j m_low_j >= (1 - alpha) pi_high) versus the
   drop-side knapsack (min sum m_hat s.t. sum b_j m_low_j >= 1 - alpha
   pi_low - sum (1 - b_j) m_low_j), keeping the smaller feasible objective.
   The resulting label sets satisfy, with probability >= 1 - delta -
   delta_m - delta_pi over the calibration draw and simultaneously for all
   classes and subsets (Theorem 4.5), class-conditional miscoverage <=
   Gamma_hat_g(S_g); Theorem 4.3 shows the population Gamma_g is the exact
   worst-case bound. Fail-closed: if both knapsacks are infeasible for a
   class, all of its intervals are kept and ``feasible`` is False (the
   certificate still holds, it is just weaker).
6. ``naive_bin_acceptance`` — the margin-free contrast baseline used by the
   tests: fixed grid bins admitted when the plain calibration mean lies in
   the bin. It has NO finite-sample validity: a single unlucky or
   adversarial calibration sample can admit a bin whose population (and
   held-out) conditional mean lies outside, which is exactly the failure
   mode Algorithm 1's margin provably controls at level delta.

Guarantee ledger (finite-sample vs asymptotic), stated precisely:
- FINITE-SAMPLE: population-mode admissibility (Theorem 2.2) — under i.i.d.
  calibration, bounded outcomes, p fixed a priori, all admitted intervals
  are population PICPIs with prob >= 1 - delta. Multiclass same-sample
  certificates (Theorem 4.5) — class-conditional miscoverage <= Gamma_hat
  with prob >= 1 - delta - delta_m - delta_pi.
- ASYMPTOTIC / RATE: Theorem 3.2's width bound holds at each n under
  lambda-regularity and n >= log(3n/delta)/(sqrt(2) lambda); the n^{-1/3}
  decay up to log factors and prediction error is a rate statement, and the
  accompanying seeded SYNTHETIC slope check illustrates it, never proves it.
- NO GUARANTEE: empirical mode (Algorithm 3) beyond the calibration sample;
  outcome-level coverage of Y (not the target of any PICPI theorem);
  anything under distribution shift between calibration and query time.

Honesty: every numeric result produced here (including the benches) is a
proper diagnostic on seeded SYNTHETIC data — self-consistency violation
rates, certified prediction-mass fractions, interval widths — never market
evidence, never Sharpe/Sortino/Calmar/P&L/NAV, and nothing here implies
live-trading capability.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]

# Grid fan-out guard: Algorithm 1 screens K(K+1)/2 candidates, so the cost is
# O(K^2); fail closed rather than allocate unbounded matrices.
MAX_GRID_BINS: int = 4096

# Integer scaling for the Algorithm-2 knapsack DP (profit granularity).
_KNAPSACK_SCALE: int = 16384


# ---------------------------------------------------------------------------
# validation / rescaling helpers
# ---------------------------------------------------------------------------


def _as_1d(values: Array, name: str) -> Array:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-d")
    return arr


def _check_range(outcome_range: tuple[float, float]) -> tuple[float, float]:
    lo, hi = float(outcome_range[0]), float(outcome_range[1])
    if not math.isfinite(lo) or not math.isfinite(hi):
        raise ValueError("outcome_range endpoints must be finite")
    if lo >= hi:
        raise ValueError("outcome_range requires lo < hi")
    return lo, hi


def _check_grid(num_bins: int) -> int:
    k = int(num_bins)
    if k != num_bins or k < 1:
        raise ValueError("num_bins must be a positive integer")
    if k > MAX_GRID_BINS:
        raise ValueError(f"num_bins must be <= {MAX_GRID_BINS} (O(K^2) screen)")
    return k


def _rescaled_pairs(
    predictions: Array,
    outcomes: Array,
    outcome_range: tuple[float, float],
) -> tuple[Array, Array, float, float]:
    """Validate calibration pairs and rescale to the unit square.

    Fail-closed: non-finite entries, out-of-range predictions or outcomes,
    length mismatch, and empty samples all raise. Predictions and outcomes
    must live on the same scale (the self-consistency condition compares
    outcome means against prediction-value intervals).
    """
    lo, hi = _check_range(outcome_range)
    p = _as_1d(predictions, "predictions")
    y = _as_1d(outcomes, "outcomes")
    if p.size != y.size:
        raise ValueError("predictions and outcomes must have the same length")
    if p.size == 0:
        raise ValueError("calibration sample must be non-empty")
    if not np.isfinite(p).all():
        raise ValueError("predictions must be finite")
    if not np.isfinite(y).all():
        raise ValueError("outcomes must be finite")
    if p.min() < lo or p.max() > hi:
        raise ValueError("predictions must lie within outcome_range")
    if y.min() < lo or y.max() > hi:
        raise ValueError("outcomes must lie within outcome_range")
    span = hi - lo
    return (p - lo) / span, (y - lo) / span, lo, span


def _sorted_prefix(zs: Array, ys: Array) -> tuple[Array, Array, Array]:
    """Sort by prediction and return (sorted predictions, cumulative-right
    counts grid helper, prefix sums of outcomes aligned to sorted order)."""
    order = np.argsort(zs, kind="mergesort")
    z_sorted = zs[order]
    y_sorted = ys[order]
    y_prefix = np.concatenate(([0.0], np.cumsum(y_sorted)))
    return z_sorted, y_sorted, y_prefix


# ---------------------------------------------------------------------------
# fit outputs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PicpiQuery:
    """Result of querying one prediction value.

    ``certified`` is True when a constructed interval was returned. Its
    meaning depends on ``PicpiFit.mode``: "population" => Theorem 2.2
    certificate (with prob >= 1 - delta over the calibration draw the
    population conditional mean lies inside); "empirical" => self-consistent
    on the calibration sample only. ``certified`` is False for the trivial
    Section 4.1 fallback [y_lo, y_hi] (population mode, when no admitted
    interval contains the value) and for uncertified regions in empirical
    mode (stall tail, or z == y_lo which the half-open bins exclude).
    """

    value: float
    lower: float
    upper: float
    certified: bool

    @property
    def width(self) -> float:
        return self.upper - self.lower


@dataclass(frozen=True)
class PicpiFit:
    """Constructed PICPI collection on the original outcome scale.

    ``lower``/``upper`` are sorted by (lower, upper). ``counts`` are
    calibration counts, ``emp_means`` the calibration outcome means, and
    ``margins`` the Hoeffding margin used at admission (Algorithm 1; the
    admitted mean lies in [lower + margin, upper - margin] up to endpoint
    relaxation). In empirical mode ``margins`` is all zeros and
    ``uncertified_from`` marks a greedy stall tail (None when the partition
    reaches y_hi).
    """

    mode: str
    lower: Array
    upper: Array
    counts: Array
    emp_means: Array
    margins: Array
    num_bins: int
    delta: float
    outcome_range: tuple[float, float]
    n_cal: int
    uncertified_from: float | None = None
    relax_endpoints: bool = False

    # -- queries -----------------------------------------------------------

    def _rescale_z(self, z: Array) -> Array:
        lo, hi = self.outcome_range
        zq = _as_1d(z, "query values")
        if zq.size == 0:
            raise ValueError("query values must be non-empty")
        if not np.isfinite(zq).all():
            raise ValueError("query values must be finite")
        if zq.min() < lo or zq.max() > hi:
            raise ValueError("query values must lie within outcome_range")
        return (zq - lo) / (hi - lo)

    def query_many(self, z: Array) -> tuple[Array, Array, NDArray[np.bool_]]:
        """Return (lower, upper, certified) per query value.

        Population mode: the shortest admitted interval containing z
        (Algorithm 1 Inference; ties by smaller lower endpoint). Empirical
        mode: the unique partition bin (t_{j-1}, t_j] containing z. The
        un-certified fallback is the trivial interval [y_lo, y_hi].
        """
        zq = self._rescale_z(z)
        n = int(zq.size)
        lo, span = self.outcome_range[0], self.outcome_range[1] - self.outcome_range[0]
        out_lo = np.full(n, float(lo))
        out_hi = np.full(n, float(lo + span))
        cert = np.zeros(n, dtype=bool)
        if self.mode == "empirical":
            if self.lower.size > 0:
                uppers = (self.upper - lo) / span
                idx = np.searchsorted(uppers, zq, side="left")
                j_count = int(self.upper.size)
                # Bins are (t_{j-1}, t_j]: z == y_lo (rescaled 0) is excluded
                # by the half-open convention; z beyond the last certified
                # bin (stall tail) gets idx == j_count. Both fall back.
                hit = (idx < j_count) & (zq > 0.0)
                sel = np.clip(idx, 0, j_count - 1)
                out_lo = np.where(hit, self.lower[sel], out_lo)
                out_hi = np.where(hit, self.upper[sel], out_hi)
                cert = hit
            return out_lo, out_hi, cert
        # population mode: shortest containing interval, first-hit by width.
        if self.lower.size == 0:
            return out_lo, out_hi, cert
        widths = self.upper - self.lower
        order = np.argsort(widths, kind="stable")
        unassigned = np.ones(n, dtype=bool)
        for k in order:
            a = float(self.lower[k])
            b = float(self.upper[k])
            hit = unassigned & (zq >= a) & (zq <= b)
            if hit.any():
                out_lo[hit] = a
                out_hi[hit] = b
                cert[hit] = True
                unassigned &= ~hit
                if not unassigned.any():
                    break
        return out_lo, out_hi, cert

    def query(self, z: float) -> PicpiQuery:
        """Single-value query; see ``query_many`` for semantics."""
        if not isinstance(z, (int, float, np.floating)) or not math.isfinite(float(z)):
            raise ValueError("query value must be a finite scalar")
        los, his, cert = self.query_many(np.array([float(z)]))
        return PicpiQuery(
            value=float(z),
            lower=float(los[0]),
            upper=float(his[0]),
            certified=bool(cert[0]),
        )

    def certified_fraction(self, z: Array) -> float:
        """Fraction of query prediction values receiving a certified interval."""
        _, _, cert = self.query_many(z)
        return float(np.mean(cert.astype(float)))

    def expand(self, z: Array, epsilon: float) -> tuple[Array, Array]:
        """Eq. (13): epsilon-expansion E^eps([l(p(x)), u(p(x))]), clipped.

        ``epsilon`` is on the original outcome scale (identical to the
        paper's unit-scale epsilon when outcome_range=(0, 1)). The expanded
        set targets the true probability p*(X) (Theorem 4.1), NOT the
        outcome Y.
        """
        eps = float(epsilon)
        if not math.isfinite(eps) or eps < 0.0:
            raise ValueError("epsilon must be finite and >= 0")
        los, his, _ = self.query_many(z)
        lo, hi = self.outcome_range
        return np.maximum(los - eps, float(lo)), np.minimum(his + eps, float(hi))

    def coverage_profile(self, z: Array, caps: Sequence[float]) -> Array:
        """Eq. (6) diagnostic: fraction of z enclosed by an admitted interval
        of width <= cap, per cap. Monotone non-decreasing in the cap; the
        complement is the excluded prediction mass (Theorem 3.2's delta_1 at
        the corresponding width bound)."""
        cap_arr = np.asarray(caps, dtype=float).ravel()
        if cap_arr.size == 0:
            raise ValueError("caps must be non-empty")
        if not np.isfinite(cap_arr).all() or cap_arr.min() < 0.0:
            raise ValueError("caps must be finite and non-negative")
        zq = self._rescale_z(z)
        out = np.zeros(int(cap_arr.size), dtype=float)
        if self.lower.size == 0:
            return out
        widths = self.upper - self.lower
        for c_idx, cap in enumerate(cap_arr):
            keep = widths <= float(cap) + 1e-12
            if not bool(keep.any()):
                continue
            a = self.lower[keep]
            b = self.upper[keep]
            inside = ((zq[:, None] >= a[None, :]) & (zq[:, None] <= b[None, :])).any(axis=1)
            out[c_idx] = float(np.mean(inside.astype(float)))
        return out


# ---------------------------------------------------------------------------
# Algorithm 1 — population mode (Theorem 2.2 validity)
# ---------------------------------------------------------------------------


def fit_picpi(
    predictions: Array,
    outcomes: Array,
    *,
    num_bins: int = 100,
    delta: float = 0.1,
    relax_endpoints: bool = False,
    outcome_range: tuple[float, float] = (0.0, 1.0),
) -> PicpiFit:
    """Algorithm 1 Calibration: grid search with Hoeffding admission margin.

    Admits candidate [i/K, j/K] when N > 0 calibration predictions fall in it
    and the empirical outcome mean lies in [a + m, b - m] with
    m = 2 sqrt(log(K^2/delta)/N) (Line 6). With probability >= 1 - delta over
    an i.i.d. calibration sample and p fixed independently, every admitted
    interval is a population PICPI (Theorem 2.2). ``relax_endpoints`` applies
    their Eq. (21) boundary relaxation. Bounded outcomes on ``outcome_range``
    are supported via the affine rescaling of Remark 3.6.
    """
    k = _check_grid(num_bins)
    d = float(delta)
    if not math.isfinite(d) or not 0.0 < d < 1.0:
        raise ValueError("delta must be in (0, 1)")
    zp, yo, lo, span = _rescaled_pairs(predictions, outcomes, outcome_range)
    n = int(zp.size)
    z_sorted, _, y_prefix = _sorted_prefix(zp, yo)
    grid = np.arange(k + 1, dtype=float) / k
    # Closed-interval counts: R[g] = #{p <= grid_g}, L[g] = #{p < grid_g}.
    right = np.searchsorted(z_sorted, grid, side="right")
    left = np.searchsorted(z_sorted, grid, side="left")
    y_right = y_prefix[right]
    y_left = y_prefix[left]
    i_idx, j_idx = np.triu_indices(k + 1, k=1)
    a = grid[i_idx]
    b = grid[j_idx]
    counts = (right[j_idx] - left[i_idx]).astype(float)
    sums = y_right[j_idx] - y_left[i_idx]
    safe = np.maximum(counts, 1.0)
    means = np.where(counts > 0.0, sums / safe, np.nan)
    log_term = math.log(float(k) * float(k) / d)
    margin_unit = 2.0 * np.sqrt(log_term / safe)
    if relax_endpoints:
        # Eq. (21): skip the margin where the mean cannot exit [0, 1] anyway.
        margin_lo = margin_unit * np.where(a > 0.0, 1.0, 0.0)
        margin_hi = margin_unit * np.where(b < 1.0, 1.0, 0.0)
    else:
        margin_lo = margin_unit
        margin_hi = margin_unit
    ok = (counts > 0.0) & (means >= a + margin_lo) & (means <= b - margin_hi)
    sel_lo = a[ok] * span + lo
    sel_hi = b[ok] * span + lo
    order = np.lexsort((sel_hi, sel_lo))
    return PicpiFit(
        mode="population",
        lower=sel_lo[order],
        upper=sel_hi[order],
        counts=counts[ok][order],
        emp_means=(means[ok] * span + lo)[order],
        margins=(margin_unit[ok] * span)[order],
        num_bins=k,
        delta=d,
        outcome_range=(lo, lo + span),
        n_cal=n,
        uncertified_from=None,
        relax_endpoints=bool(relax_endpoints),
    )


# ---------------------------------------------------------------------------
# Algorithm 3 — empirical mode (self-consistency on the calibration sample)
# ---------------------------------------------------------------------------


def fit_picpi_empirical(
    predictions: Array,
    outcomes: Array,
    *,
    num_bins: int = 100,
    outcome_range: tuple[float, float] = (0.0, 1.0),
) -> PicpiFit:
    """Algorithm 3: greedy self-consistent partition (empirical mode).

    Bins are (t_{j-1}, t_j] on the grid {0, 1/K, ..., 1}; each admitted bin
    has empirical outcome mean inside itself AND leaves a remainder whose
    empirical mean is inside the remainder (Line 5). NO population validity
    (Section 5.4 states the margin omission forfeits Theorem 2.2).
    Fail-closed edges: empty bins are never admitted; if no endpoint
    qualifies, the tail after the last admitted bin is uncertified
    (``uncertified_from``) and queries there fall back to [y_lo, y_hi].
    """
    k = _check_grid(num_bins)
    zp, yo, lo, span = _rescaled_pairs(predictions, outcomes, outcome_range)
    n = int(zp.size)
    z_sorted, _, y_prefix = _sorted_prefix(zp, yo)
    grid = np.arange(k + 1, dtype=float) / k
    right = np.searchsorted(z_sorted, grid, side="right")
    y_right = y_prefix[right]
    y_total = float(y_prefix[n])
    lowers: list[float] = []
    uppers: list[float] = []
    counts_list: list[float] = []
    means_list: list[float] = []
    uncertified_from: float | None = None
    t_idx = 0
    while t_idx < k:
        js = np.arange(t_idx + 1, k + 1)
        g = grid[js]
        cnt = (right[js] - right[t_idx]).astype(float)
        sm = y_right[js] - y_right[t_idx]
        safe_cnt = np.maximum(cnt, 1.0)
        bin_mean = np.where(cnt > 0.0, sm / safe_cnt, np.nan)
        rem_cnt = float(n) - right[js].astype(float)
        rem_mean = np.where(
            rem_cnt > 0.0, (y_total - y_right[js]) / np.maximum(rem_cnt, 1.0), np.nan
        )
        bin_ok = (cnt > 0.0) & (bin_mean > grid[t_idx]) & (bin_mean <= g)
        rem_ok = (js == k) | ((rem_cnt > 0.0) & (rem_mean > g) & (rem_mean <= 1.0))
        valid = bin_ok & rem_ok
        if not bool(valid.any()):
            # Stall: the remainder cannot be certified even as a single bin.
            uncertified_from = float(grid[t_idx]) * span + lo
            break
        j_sel = int(js[int(np.argmax(valid))])
        cnt_sel = float(right[j_sel] - right[t_idx])
        mean_sel = float((y_right[j_sel] - y_right[t_idx]) / cnt_sel)
        lowers.append(float(grid[t_idx]) * span + lo)
        uppers.append(float(grid[j_sel]) * span + lo)
        counts_list.append(cnt_sel)
        means_list.append(mean_sel * span + lo)
        if j_sel == k:
            break
        t_idx = j_sel
    return PicpiFit(
        mode="empirical",
        lower=np.asarray(lowers, dtype=float),
        upper=np.asarray(uppers, dtype=float),
        counts=np.asarray(counts_list, dtype=float),
        emp_means=np.asarray(means_list, dtype=float),
        margins=np.zeros(len(lowers), dtype=float),
        num_bins=k,
        delta=float("nan"),
        outcome_range=(lo, lo + span),
        n_cal=n,
        uncertified_from=uncertified_from,
        relax_endpoints=False,
    )


# ---------------------------------------------------------------------------
# Algorithm 4 — DisjointCalibration (Appendix C)
# ---------------------------------------------------------------------------


def disjoint_calibration(
    lower: Array,
    upper: Array,
    max_length: float,
) -> tuple[Array, Array]:
    """Algorithm 4: length filter + maximum-covered-length disjoint subset.

    Weighted interval scheduling with weight = interval length (their
    MaxCoverageIntervals): sort by right endpoint, DP over compatible
    predecessors, backtrack. Returns the selected intervals sorted by lower
    endpoint; empty arrays when nothing passes the length filter (their
    Line 10).
    """
    lo_arr = np.asarray(lower, dtype=float).ravel()
    up_arr = np.asarray(upper, dtype=float).ravel()
    if lo_arr.size != up_arr.size:
        raise ValueError("lower and upper must have the same length")
    if lo_arr.size and (not np.isfinite(lo_arr).all() or not np.isfinite(up_arr).all()):
        raise ValueError("interval endpoints must be finite")
    if lo_arr.size and bool((up_arr < lo_arr).any()):
        raise ValueError("upper endpoints must be >= lower endpoints")
    ell = float(max_length)
    if not math.isfinite(ell) or ell < 0.0:
        raise ValueError("max_length must be finite and >= 0")
    keep = (up_arr - lo_arr) <= ell + 1e-12
    s = lo_arr[keep]
    e = up_arr[keep]
    m = int(s.size)
    if m == 0:
        return np.zeros(0, dtype=float), np.zeros(0, dtype=float)
    order = np.lexsort((s, e))
    s, e = s[order], e[order]
    w = e - s
    # p(j): number of intervals ending at or before s_j (equals the OPT index
    # of the last compatible predecessor in their 1-based DP).
    compat = np.searchsorted(e, s, side="right")
    opt = np.zeros(m + 1, dtype=float)
    for j in range(1, m + 1):
        take = float(w[j - 1]) + opt[int(compat[j - 1])]
        opt[j] = max(take, opt[j - 1])
    chosen: list[int] = []
    j = m
    while j > 0:
        take = float(w[j - 1]) + opt[int(compat[j - 1])]
        if take > opt[j - 1]:
            chosen.append(j - 1)
            j = int(compat[j - 1])
        else:
            j -= 1
    sel = np.asarray(sorted(chosen), dtype=int)
    return s[sel].copy(), e[sel].copy()


# ---------------------------------------------------------------------------
# naive contrast baseline (no validity — documented in module docstring)
# ---------------------------------------------------------------------------


def naive_bin_acceptance(
    predictions: Array,
    outcomes: Array,
    *,
    num_bins: int = 100,
    outcome_range: tuple[float, float] = (0.0, 1.0),
) -> tuple[Array, Array]:
    """Margin-free baseline: fixed grid bins whose calibration mean lies inside.

    This is what Algorithm 1 reduces to with the concentration margin set to
    zero, restricted to single-step bins. It carries NO finite-sample
    guarantee (Theorem 2.2 does not apply): admitted bins can have population
    conditional means outside the bin. Present only as a contrast baseline.
    """
    k = _check_grid(num_bins)
    zp, yo, lo, span = _rescaled_pairs(predictions, outcomes, outcome_range)
    z_sorted, _, y_prefix = _sorted_prefix(zp, yo)
    grid = np.arange(k + 1, dtype=float) / k
    right = np.searchsorted(z_sorted, grid, side="right")
    left = np.searchsorted(z_sorted, grid, side="left")
    y_right = y_prefix[right]
    y_left = y_prefix[left]
    counts = (right[1:] - left[:-1]).astype(float)
    sums = y_right[1:] - y_left[:-1]
    means = np.where(counts > 0.0, sums / np.maximum(counts, 1.0), np.nan)
    a = grid[:-1]
    b = grid[1:]
    ok = (counts > 0.0) & (means >= a) & (means <= b)
    return a[ok] * span + lo, b[ok] * span + lo


# ---------------------------------------------------------------------------
# held-out self-consistency diagnostics
# ---------------------------------------------------------------------------


def holdout_self_consistency(
    lower: Array,
    upper: Array,
    predictions: Array,
    outcomes: Array,
    *,
    min_count: int = 200,
    outcome_range: tuple[float, float] = (0.0, 1.0),
) -> dict[str, float]:
    """Fraction of intervals whose held-out outcome mean exits the interval.

    A SYNTHETIC sanity diagnostic, not a guarantee: even genuine population
    PICPIs show empirical held-out excursions at the sqrt(1/N_held) noise
    scale, so ``min_count`` gates the check and small violation rates are
    expected by construction. For margin-free families (naive bins) the same
    statistic exposes the absence of any validity control.
    """
    lo_arr = np.asarray(lower, dtype=float).ravel()
    up_arr = np.asarray(upper, dtype=float).ravel()
    if lo_arr.size != up_arr.size:
        raise ValueError("lower and upper must have the same length")
    mc = int(min_count)
    if mc < 1:
        raise ValueError("min_count must be >= 1")
    if lo_arr.size == 0:
        return {"violation_rate": 0.0, "n_checked": 0.0, "max_violation_distance": 0.0}
    _, yo, lo, span = _rescaled_pairs(predictions, outcomes, outcome_range)
    zp = _as_1d(predictions, "predictions")
    zq = (zp - lo) / span
    a = (lo_arr - lo) / span
    b = (up_arr - lo) / span
    inside = (zq[:, None] >= a[None, :]) & (zq[:, None] <= b[None, :])
    counts = inside.sum(axis=0).astype(float)
    sums = yo @ inside.astype(float)
    checked = counts >= float(mc)
    if not bool(checked.any()):
        return {"violation_rate": 0.0, "n_checked": 0.0, "max_violation_distance": 0.0}
    means = sums[checked] / counts[checked]
    aa = a[checked]
    bb = b[checked]
    dist = np.maximum(np.maximum(aa - means, means - bb), 0.0)
    viol = dist > 1e-12
    return {
        "violation_rate": float(np.mean(viol.astype(float))),
        "n_checked": float(int(checked.sum())),
        "max_violation_distance": float(dist.max()) if bool(viol.any()) else 0.0,
    }


# ---------------------------------------------------------------------------
# Section 4.2 — multiclass label sets (Algorithms 2 + 4, Theorems 4.3/4.5)
# ---------------------------------------------------------------------------


def _min_weight_cover(profits: Array, weights: Array, target: float) -> NDArray[np.bool_] | None:
    """Exact 0/1 knapsack: min sum(weights) s.t. sum(profits) >= target.

    Used for both Algorithm-2 subproblems (Eqs. (19)-(20) and their
    Theorem-4.5 empirical relaxation): keep-side profits a_j m_j, drop-side
    profits b_j m_j, weights m_hat_j. Integer-scaled DP over covered profit
    levels; flooring the scaled profits and ceiling the target is
    conservative (can only declare infeasible, never admits a set that
    misses the true target). Returns a boolean mask, or None when infeasible.
    """
    pr = np.asarray(profits, dtype=float).ravel()
    wt = np.asarray(weights, dtype=float).ravel()
    if pr.size != wt.size:
        raise ValueError("profits and weights must have the same length")
    if pr.size and (not np.isfinite(pr).all() or not np.isfinite(wt).all()):
        raise ValueError("profits and weights must be finite")
    if pr.size and bool((pr < 0.0).any() or (wt < 0.0).any()):
        raise ValueError("profits and weights must be non-negative")
    tgt = float(target)
    if tgt <= 0.0:
        return np.zeros(int(pr.size), dtype=bool)
    if pr.size == 0 or float(pr.sum()) < tgt:
        return None
    scale = float(_KNAPSACK_SCALE) / float(pr.sum())
    ip = np.floor(pr * scale).astype(np.int64)
    keep = ip > 0
    if not bool(keep.any()):
        return None
    ip_k = ip[keep]
    wt_k = wt[keep]
    level = int(min(math.ceil(tgt * scale), int(ip_k.sum())))
    if level <= 0:
        return np.zeros(int(pr.size), dtype=bool)
    inf = float("inf")
    dp = np.full(level + 1, inf, dtype=float)
    dp[0] = 0.0
    states = np.arange(level + 1)
    snaps: list[Array] = [dp.copy()]
    for j in range(int(ip_k.size)):
        src = np.maximum(states - int(ip_k[j]), 0)
        dp = np.minimum(dp, dp[src] + float(wt_k[j]))
        snaps.append(dp.copy())
    if not math.isfinite(float(dp[level])):
        return None
    keep_positions = np.flatnonzero(keep)
    mask = np.zeros(int(pr.size), dtype=bool)
    c = level
    for pos in range(int(ip_k.size) - 1, -1, -1):
        if c <= 0:
            break
        src_c = max(c - int(ip_k[pos]), 0)
        if snaps[pos][src_c] + float(wt_k[pos]) == snaps[pos + 1][c]:
            mask[int(keep_positions[pos])] = True
            c = src_c
    return mask


@dataclass(frozen=True)
class LabelSetRule:
    """Per-class retained-interval subsets from the Algorithm-2 solver.

    ``keep``[g] is a boolean mask over class g's disjoint intervals;
    ``gamma_hat``[g] is the Theorem-4.5 empirical miscoverage certificate for
    the retained subset; ``feasible``[g] is False when neither knapsack met
    the target alpha_g (fail-closed: all intervals kept, certificate weaker).
    """

    keep: tuple[NDArray[np.bool_], ...]
    gamma_hat: tuple[float, ...]
    feasible: tuple[bool, ...]

    def __post_init__(self) -> None:
        # Guard against accidental direct construction misuse.
        if len(self.keep) != len(self.gamma_hat) or len(self.keep) != len(self.feasible):
            raise ValueError("keep, gamma_hat, feasible must have equal length")


@dataclass(frozen=True)
class MulticlassPicpi:
    """Per-class disjoint PICPI tables plus Theorem-4.5 empirical quantities.

    ``lower``/``upper``/``masses`` are per-class arrays over the disjoint
    intervals returned by Algorithm 4 (masses = empirical prediction mass in
    each interval on the SAME calibration sample); ``priors`` are empirical
    class frequencies; ``eps_m``/``eps_pi`` are the Hoeffding radii of the
    Section-4.2 empirical relaxation with N_cand = G K(K+1)/2 candidates.
    """

    lower: tuple[Array, ...]
    upper: tuple[Array, ...]
    masses: tuple[Array, ...]
    priors: tuple[float, ...]
    eps_m: float
    eps_pi: float
    n_cal: int
    n_classes: int
    num_bins: int
    delta: float
    delta_m: float
    delta_pi: float

    def _class_bounds(self, g: int) -> tuple[Array, Array, Array, float, float]:
        m_low = np.maximum(self.masses[g] - self.eps_m, 0.0)
        pi_low = max(self.priors[g] - self.eps_pi, 0.0)
        pi_high = min(self.priors[g] + self.eps_pi, 1.0)
        return self.lower[g], self.upper[g], m_low, pi_low, pi_high

    def gamma_hat(self, g: int, keep: NDArray[np.bool_]) -> tuple[float, float, float]:
        """Theorem-4.5 certificate: (Gamma_up, Gamma_low, min) for a subset."""
        if not 0 <= g < self.n_classes:
            raise ValueError("class index out of range")
        mask = np.asarray(keep).astype(bool).ravel()
        if mask.size != self.lower[g].size:
            raise ValueError("keep mask must match the class interval count")
        a, b, m_low, pi_low, pi_high = self._class_bounds(g)
        if pi_low > 0.0:
            gamma_up = (
                1.0 - float(np.sum(m_low[mask])) - float(np.sum((1.0 - b[~mask]) * m_low[~mask]))
            ) / pi_low
        else:
            gamma_up = float("inf")
        gamma_low = 1.0 - float(np.sum(a[mask] * m_low[mask])) / pi_high if pi_high > 0.0 else 1.0
        return gamma_up, gamma_low, min(gamma_up, gamma_low)

    def select(self, alphas: Sequence[float]) -> LabelSetRule:
        """Empirical Algorithm 2: per-class keep/drop knapsacks at level alpha_g."""
        alpha_arr = np.asarray(alphas, dtype=float).ravel()
        if alpha_arr.size != self.n_classes:
            raise ValueError("alphas must have one entry per class")
        if not np.isfinite(alpha_arr).all() or alpha_arr.min() < 0.0 or alpha_arr.max() >= 1.0:
            raise ValueError("alphas must lie in [0, 1)")
        keeps: list[NDArray[np.bool_]] = []
        gammas: list[float] = []
        feasibles: list[bool] = []
        for g in range(self.n_classes):
            a, b, m_low, pi_low, pi_high = self._class_bounds(g)
            mhat = self.masses[g]
            alpha_g = float(alpha_arr[g])
            j_size = int(a.size)
            # Keep side: Gamma_low <= alpha  <=>  sum_S a_j m_low_j >= (1-alpha) pi_high.
            keep_mask = _min_weight_cover(a * m_low, mhat, (1.0 - alpha_g) * pi_high)
            keep_obj = float(np.sum(mhat[keep_mask])) if keep_mask is not None else float("inf")
            # Drop side: Gamma_up <= alpha  <=>  sum_S b_j m_low_j >= RHS.
            rhs = 1.0 - alpha_g * pi_low - float(np.sum((1.0 - b) * m_low))
            drop_mask = _min_weight_cover(b * m_low, mhat, rhs) if pi_low > 0.0 else None
            drop_obj = float(np.sum(mhat[drop_mask])) if drop_mask is not None else float("inf")
            if keep_mask is not None and keep_obj <= drop_obj:
                mask, feasible = keep_mask, True
            elif drop_mask is not None:
                mask, feasible = drop_mask, True
            else:
                # Fail-closed fallback: keep every interval; certificate stays
                # valid (Theorem 4.5 holds for ALL subsets), just weaker.
                mask = np.ones(j_size, dtype=bool)
                feasible = False
            keeps.append(mask)
            gammas.append(self.gamma_hat(g, mask)[2])
            feasibles.append(feasible)
        return LabelSetRule(keep=tuple(keeps), gamma_hat=tuple(gammas), feasible=tuple(feasibles))

    def predict(self, rule: LabelSetRule, probabilities: Array) -> NDArray[np.bool_]:
        """Eq. (14): label set = classes whose p_g(x) falls in a kept interval."""
        probs = np.asarray(probabilities, dtype=float)
        if probs.ndim != 2:
            raise ValueError("probabilities must be a 2-d (n, G) array")
        n, g_cols = int(probs.shape[0]), int(probs.shape[1])
        if g_cols != self.n_classes:
            raise ValueError("probabilities must have one column per class")
        if n == 0:
            raise ValueError("probabilities must contain at least one row")
        if not np.isfinite(probs).all():
            raise ValueError("probabilities must be finite")
        if probs.min() < 0.0 or probs.max() > 1.0:
            raise ValueError("probabilities must lie in [0, 1]")
        if len(rule.keep) != self.n_classes:
            raise ValueError("rule does not match this fit")
        out = np.zeros((n, self.n_classes), dtype=bool)
        for g in range(self.n_classes):
            mask = rule.keep[g]
            if not bool(mask.any()):
                continue
            a = self.lower[g][mask]
            b = self.upper[g][mask]
            col = probs[:, g]
            out[:, g] = ((col[:, None] >= a[None, :]) & (col[:, None] <= b[None, :])).any(axis=1)
        return out


def fit_picpi_multiclass(
    probabilities: Array,
    labels: Array,
    *,
    num_bins: int = 100,
    delta: float = 0.1,
    delta_m: float = 0.05,
    delta_pi: float = 0.05,
    max_interval_length: float = 1.0,
) -> MulticlassPicpi:
    """Section 4.2 construction: per-class one-vs-rest PICPIs, disjointified.

    Each class runs Algorithm 1 on (p_g, 1{Y = g}) with failure budget
    delta / G (Theorem 4.5 requires classwise budgets summing to delta_PI =
    delta), then Algorithm 4 with ``max_interval_length``. Masses and priors
    are estimated on the SAME calibration sample with Hoeffding radii eps_m
    (union bound over all G K(K+1)/2 searched candidates) and eps_pi. Every
    class must appear at least once in the calibration labels (fail-closed:
    pi_g > 0 is assumed throughout Section 4.2).
    """
    k = _check_grid(num_bins)
    d = float(delta)
    d_m = float(delta_m)
    d_pi = float(delta_pi)
    for name, val in (("delta", d), ("delta_m", d_m), ("delta_pi", d_pi)):
        if not math.isfinite(val) or not 0.0 < val < 1.0:
            raise ValueError(f"{name} must be in (0, 1)")
    ell = float(max_interval_length)
    if not math.isfinite(ell) or ell <= 0.0:
        raise ValueError("max_interval_length must be finite and > 0")
    probs = np.asarray(probabilities, dtype=float)
    if probs.ndim != 2:
        raise ValueError("probabilities must be a 2-d (n, G) array")
    n, g_cols = int(probs.shape[0]), int(probs.shape[1])
    if n == 0 or g_cols == 0:
        raise ValueError("probabilities must be non-empty")
    if not np.isfinite(probs).all():
        raise ValueError("probabilities must be finite")
    if probs.min() < 0.0 or probs.max() > 1.0:
        raise ValueError("probabilities must lie in [0, 1]")
    lab = np.asarray(labels)
    if lab.shape != (n,):
        raise ValueError("labels must align with probabilities rows")
    if not np.issubdtype(lab.dtype, np.integer):
        raise ValueError("labels must be integers")
    if lab.min() < 0 or lab.max() >= g_cols:
        raise ValueError("labels must lie in [0, G-1]")
    lowers: list[Array] = []
    uppers: list[Array] = []
    masses: list[Array] = []
    priors: list[float] = []
    for g in range(g_cols):
        y_g = (lab == g).astype(float)
        if float(y_g.sum()) == 0.0:
            raise ValueError(f"class {g} absent from calibration labels (pi_g > 0 required)")
        fit_g = fit_picpi(probs[:, g], y_g, num_bins=k, delta=d / float(g_cols))
        dl, du = disjoint_calibration(fit_g.lower, fit_g.upper, ell)
        if dl.size:
            p_sorted = np.sort(probs[:, g])
            cnt = (
                np.searchsorted(p_sorted, du, side="right")
                - np.searchsorted(p_sorted, dl, side="left")
            ).astype(float)
        else:
            cnt = np.zeros(0, dtype=float)
        lowers.append(dl)
        uppers.append(du)
        masses.append(cnt / float(n))
        priors.append(float(np.mean(y_g)))
    n_cand = float(g_cols) * float(k) * (float(k) + 1.0) / 2.0
    eps_m = math.sqrt(math.log(2.0 * n_cand / d_m) / (2.0 * float(n)))
    eps_pi = math.sqrt(math.log(2.0 * float(g_cols) / d_pi) / (2.0 * float(n)))
    return MulticlassPicpi(
        lower=tuple(lowers),
        upper=tuple(uppers),
        masses=tuple(masses),
        priors=tuple(priors),
        eps_m=float(eps_m),
        eps_pi=float(eps_pi),
        n_cal=n,
        n_classes=g_cols,
        num_bins=k,
        delta=d,
        delta_m=d_m,
        delta_pi=d_pi,
    )


# ---------------------------------------------------------------------------
# SYNTHETIC data + benches (correctness diagnostics, never market evidence)
# ---------------------------------------------------------------------------


def synthetic_picpi_data(
    n_cal: int,
    n_test: int,
    seed: int,
    *,
    spike_weight: float = 0.0,
    spike_value: float = 0.28,
    spike_rate: float = 0.45,
) -> tuple[Array, Array, Array, Array]:
    """Seeded SYNTHETIC binary data: (p_cal, y_cal, p_test, y_test).

    Background: p ~ Uniform[0, 1] (1-regular in the sense of Definition 3.1),
    Y | p ~ Bernoulli(p) — perfectly calibrated, prediction error eps = 0.
    Optional adversarial atom: with probability ``spike_weight`` the
    prediction collapses to ``spike_value`` while the outcome rate is
    ``spike_rate`` != spike_value (a miscalibrated prediction spike).
    Deterministic given the seed (PCG64 streams are platform-stable).
    """
    n_c = int(n_cal)
    n_t = int(n_test)
    if n_c < 1 or n_t < 1:
        raise ValueError("n_cal and n_test must be >= 1")
    w = float(spike_weight)
    if not math.isfinite(w) or w < 0.0 or w >= 1.0:
        raise ValueError("spike_weight must be in [0, 1)")
    v = float(spike_value)
    r = float(spike_rate)
    if not math.isfinite(v) or not 0.0 <= v <= 1.0:
        raise ValueError("spike_value must lie in [0, 1]")
    if not math.isfinite(r) or not 0.0 <= r <= 1.0:
        raise ValueError("spike_rate must lie in [0, 1]")
    rng = np.random.default_rng(int(seed))
    n = n_c + n_t
    is_spike = rng.random(n) < w
    base = rng.random(n)
    draws = rng.random(n)
    p = np.where(is_spike, v, base)
    y = (draws < np.where(is_spike, r, p)).astype(float)
    return p[:n_c], y[:n_c], p[n_c:], y[n_c:]


def bench_picpi(
    *,
    p_cal: Array | None = None,
    y_cal: Array | None = None,
    p_test: Array | None = None,
    y_test: Array | None = None,
    num_bins: int = 20,
    delta: float = 0.1,
    epsilon_expand: float = 0.0,
    min_count: int = 200,
    n_cal: int = 20_000,
    n_test: int = 20_000,
    spike_weight: float = 0.0,
    seed: int = 7,
    dgp: str | None = None,
) -> dict[str, float | str]:
    """PICPI research diagnostics (proper scores only; SYNTHETIC fixture path).

    Pass all four panel arrays or none; with none, the seeded
    ``synthetic_picpi_data`` fixture is used (``dgp='fixture'``). Keys are
    calibration-quality diagnostics — self-consistency violation rates on
    held-out data, certified prediction-mass fraction, interval widths —
    never Sharpe/P&L style performance claims.
    """
    panel = (p_cal, y_cal, p_test, y_test)
    if any(a is not None for a in panel):
        if any(a is None for a in panel):
            raise ValueError("pass all panel cal/test arrays or none")
        p_c = np.asarray(p_cal, dtype=float)
        y_c = np.asarray(y_cal, dtype=float)
        p_t = np.asarray(p_test, dtype=float)
        y_t = np.asarray(y_test, dtype=float)
        dgp_label = dgp or "panel"
    else:
        p_c, y_c, p_t, y_t = synthetic_picpi_data(n_cal, n_test, seed, spike_weight=spike_weight)
        dgp_label = "fixture"
    fit = fit_picpi(p_c, y_c, num_bins=num_bins, delta=delta)
    los, his, cert = fit.query_many(p_t)
    widths = his - los
    report = holdout_self_consistency(fit.lower, fit.upper, p_t, y_t, min_count=min_count)
    naive_lo, naive_hi = naive_bin_acceptance(p_c, y_c, num_bins=num_bins)
    naive_report = holdout_self_consistency(naive_lo, naive_hi, p_t, y_t, min_count=min_count)
    eps = float(epsilon_expand)
    exp_lo, exp_hi = fit.expand(p_t, eps)
    out: dict[str, float | str] = {
        "certified_fraction": float(np.mean(cert.astype(float))),
        "mean_width": float(np.mean(widths[cert])) if bool(cert.any()) else float("nan"),
        "median_width": float(np.median(widths[cert])) if bool(cert.any()) else float("nan"),
        "mean_expanded_width": float(np.mean(exp_hi - exp_lo)),
        "selfconsistency_violation_rate": report["violation_rate"],
        "selfconsistency_n_checked": report["n_checked"],
        "selfconsistency_max_violation_distance": report["max_violation_distance"],
        "naive_violation_rate": naive_report["violation_rate"],
        "naive_n_checked": naive_report["n_checked"],
        "naive_n_intervals": float(naive_lo.size),
        "n_intervals": float(fit.lower.size),
        "n": float(p_t.size),
        "num_bins": float(num_bins),
        "delta": float(delta),
        "epsilon_expand": eps,
        "dgp": dgp_label,
        "claim": "research_metric_only",
    }
    if dgp_label == "fixture":
        out["seed"] = float(seed)
        out["spike_weight"] = float(spike_weight)
    return out


def bench_picpi_width_rate(
    n_grid: Sequence[int] = (10_000, 40_000, 160_000, 640_000),
    *,
    num_bins: int = 100,
    delta: float = 0.1,
    n_test: int = 2_000,
    seed: int = 11,
) -> dict[str, float | str]:
    """SYNTHETIC illustration of Theorem 3.2's n^{-1/3} width rate.

    Follows the paper's Appendix B.5 protocol on the calibrated fixture:
    Algorithm 1 with ``num_bins`` grid points and failure level ``delta`` at
    each n; per held-out prediction take the minimal width among admitted
    intervals containing it, assigning width 1.0 (the trivial fallback) when
    none does; average. The reported slope is an OLS fit of log(mean width)
    on log(n) over the geometric ``n_grid`` — an ILLUSTRATION of the rate on
    seeded synthetic data (deterministic given the seed), not a proof, and
    the fixed grid floors widths at 1/K so the slope flattens once the
    theoretical width approaches the grid step.
    """
    ns = [int(v) for v in n_grid]
    if len(ns) < 3:
        raise ValueError("n_grid must contain at least 3 sizes")
    if any(v < 100 for v in ns):
        raise ValueError("n_grid sizes must be >= 100")
    k = _check_grid(num_bins)
    d = float(delta)
    if not math.isfinite(d) or not 0.0 < d < 1.0:
        raise ValueError("delta must be in (0, 1)")
    widths: list[float] = []
    certified: list[float] = []
    for idx, n in enumerate(ns):
        p_c, y_c, p_t, _ = synthetic_picpi_data(n, int(n_test), int(seed) + 1_000 * idx)
        fit = fit_picpi(p_c, y_c, num_bins=k, delta=d)
        cap = np.full(int(p_t.size), 1.0, dtype=float)
        if fit.lower.size:
            w_arr = fit.upper - fit.lower
            order = np.argsort(w_arr, kind="stable")
            unassigned = np.ones(int(p_t.size), dtype=bool)
            for j in order:
                a = float(fit.lower[j])
                b = float(fit.upper[j])
                hit = unassigned & (p_t >= a) & (p_t <= b)
                if hit.any():
                    cap[hit] = b - a
                    unassigned &= ~hit
                    if not unassigned.any():
                        break
        widths.append(float(np.mean(cap)))
        certified.append(float(np.mean((~unassigned).astype(float))))
    log_n = np.log(np.asarray(ns, dtype=float))
    log_w = np.log(np.asarray(widths, dtype=float))
    slope, intercept = np.polyfit(log_n, log_w, 1)
    out: dict[str, float | str] = {
        "width_slope": float(slope),
        "width_intercept": float(intercept),
        "width_first": widths[0],
        "width_last": widths[-1],
        "certified_fraction_last": certified[-1],
        "n_points": float(len(ns)),
        "n_min": float(min(ns)),
        "n_max": float(max(ns)),
        "num_bins": float(k),
        "delta": d,
        "dgp": "fixture",
        "claim": "research_metric_only",
        "seed": float(seed),
    }
    return out
