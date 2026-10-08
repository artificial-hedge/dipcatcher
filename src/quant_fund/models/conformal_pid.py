"""Conformal PID control: online quantile-level control for conformal coverage (SYNTHETIC).

Proportional-integral-derivative (PID) control of the coverage level
alpha_t, closing the loop between realized miscoverage and the conformal
quantile that sets the band width. The controller treats the prediction-set
machinery as a dynamical system to be regulated: the level alpha_t is the
control signal, err_t = 1{y_t outside the band at level alpha_t} is the
measured output, and alpha is the set point (Angelopoulos, Candès, Tibshirani,
2023, "Conformal PID Control for Time Series Prediction", NeurIPS 36,
pp. 23047-23074, arXiv:2307.16895 — their Eq. (3) applies the same PID
decomposition to the score quantile q_t; here, as in the ACI recursion of
Gibbs & Candès, 2021, NeurIPS 34, pp. 1660-1672, arXiv:2106.00170, the state
being regulated is the coverage level alpha_t itself, the parametrization
used by the alpha-level PID controllers in the follow-up literature).

Update (per step, after observing err_t, with e_t = alpha - err_t):

    alpha_{t+1} = clip(alpha_t + Ki * e_t
                              + Kp * (e_t - e_{t-1})
                              + Kd * (e_t - 2 e_{t-1} + e_{t-2}), clip)

- I term: summed over t it telescopes the P and D contributions, so long-run
  coverage (1/T) sum_t err_t -> alpha is governed by the integral path, in
  exact analogy with the saturating integrator r_t of Angelopoulos et al.
  (2023, Eq. (5) and Theorem 1: long-run coverage holds for ANY scorecaster
  when the integrator satisfies their saturation condition (4)). With
  Ki = 0 the level has bounded total variation |alpha_t - alpha_0| <= Kp * 1
  (the difference term telescopes), so a pure proportional loop reacts to
  *changes* in err but has no restoring force against a persistent
  miscoverage rate — the integral term is what locks in coverage.
- P term: reacts to the innovation in the error signal (a miss arriving or
  clearing), giving a faster transient than ACI at the same effective
  integral gain; this is the role of the derivative-style kick in the paper's
  Eq. (3), where g'_t = g_t - g_{t-1} is their D channel.
- D term (Kd > 0): second difference of e, off by default.

Tuning (Angelopoulos et al., 2023, App. B): the integral gain sets the
tracking time constant — a scale of O(1/sqrt(T)) for a horizon of T steps
keeps the per-step level movement O(1/sqrt(T)) while still correcting
accumulated coverage error; the P gain is kept small relative to Ki. Defaults
here: Ki = 1/sqrt(1000) (reference horizon T = 1000), Kp = Ki/6, Kd = 0.
The paper's tangent integrator r_t(x) = K_I tan(x log t / (t C_sat)) with
C_sat = (2/pi)(ceil(log(T) delta) - 1/log(T)) and K_I set to the
hypothesized score bound is the quantile-space analogue; in level space the
clip box plays the saturation role.

Two-sided vs one-sided treatment (Angelopoulos et al., 2023, Sec. 3):
``ConformalPID`` is agnostic to the band construction — the caller supplies
err_t for any band (two-sided central band or one-sided tail bound).
``ConformalPIDQuantile`` drives an actual quantile grid: every level tau_j is
conformalized with the one-sided upper-coverage score s_t = y_t - q_tau_j
and miss indicator err_{t,j} = 1{y_t > q̃_{t,j}}, targeting P(y <= q̃_j) ->
tau_j; a two-sided central band [q̃_{alpha/2}, q̃_{1-alpha/2}] is then the
pair of tail levels, whose miscoverage budget alpha_t is tracked by the two
corresponding level loops. Output grids are made non-crossing by a
cumulative maximum followed by row-wise rearrangement (Chernozhukov,
Fernández-Val, Galichon, 2010, Ann. Statist. 38; the
quant_fund.metrics.scoring.rearrange_quantiles convention shared with
quant_fund.models.agaci).

Honesty: only coverage, miscoverage levels, and pinball-consistent quantile
grids are exposed (AGENTS.md honesty contract). Long-run coverage is a
time-average guarantee, not a finite-horizon probabilistic one.

Conventions: numpy core, fail-closed edges (ValueError), seeded tie-breaking
is not required — the controller is deterministic given the error stream.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from quant_fund.metrics.conformal import conformal_quantile
from quant_fund.metrics.scoring import rearrange_quantiles

Array = NDArray[np.float64]

__all__ = ["ConformalPID", "ConformalPIDQuantile"]

# Reference horizon for the default integral gain: Ki = 1/sqrt(_REF_T),
# the O(1/sqrt(T)) scale suggested in Angelopoulos et al. 2023, App. B.
_REF_T = 1000.0


def _check_alpha(alpha: float) -> float:
    a = float(alpha)
    if not 0.0 < a < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    return a


def _check_gain(name: str, value: float | None, default: float) -> float:
    g = default if value is None else float(value)
    if not np.isfinite(g) or g < 0.0:
        raise ValueError(f"{name} must be non-negative and finite")
    return g


def _check_clip(clip: tuple[float, float]) -> tuple[float, float]:
    lo, hi = float(clip[0]), float(clip[1])
    if not (np.isfinite(lo) and np.isfinite(hi)) or not 0.0 < lo < hi < 1.0:
        raise ValueError("clip must satisfy 0 < lo < hi < 1")
    return lo, hi


def _check_levels(quantile_levels: Array | None, alpha: float) -> Array:
    if quantile_levels is None:
        return np.array([alpha / 2.0, 1.0 - alpha / 2.0], dtype=float)
    levels = np.asarray(quantile_levels, dtype=float).reshape(-1)
    if levels.size == 0:
        raise ValueError("quantile_levels must be non-empty")
    if not np.all(np.isfinite(levels)) or not np.all((levels > 0.0) & (levels < 1.0)):
        raise ValueError("quantile_levels must lie in (0, 1)")
    if np.any(np.diff(levels) <= 0.0):
        raise ValueError("quantile_levels must be strictly increasing")
    return levels


def _pid_step(
    alpha_t: Array,
    e: Array,
    e_prev: Array,
    e_prev2: Array,
    *,
    ki: float,
    kp: float,
    kd: float,
    lo: float,
    hi: float,
) -> Array:
    """One vectorized PID level update; see module docstring for the recursion."""
    nxt = alpha_t + ki * e + kp * (e - e_prev) + kd * (e - 2.0 * e_prev + e_prev2)
    return np.clip(nxt, lo, hi)


class ConformalPID:
    """Scalar PID controller of a conformal coverage level.

    Parameters
    ----------
    alpha:
        Target miscoverage in (0, 1); the set point of the loop.
    Ki, Kp, Kd:
        Integral, proportional, and derivative gains; non-negative and
        finite. None -> Ki = 1/sqrt(1000), Kp = Ki/6 (the O(1/sqrt(T))
        integral scale with a small proportional gain, per Angelopoulos et
        al., 2023, App. B). Kd defaults to 0 (disabled).
    clip:
        Saturation box (lo, hi) with 0 < lo < hi < 1 applied to alpha_t
        after every update; plays the role of the saturating integrator in
        the paper's Theorem 1.
    quantile_levels:
        Optional validated record of the quantile grid the caller's band is
        built from (informational; the scalar loop only sees err_t).

    Notes
    -----
    ``update(err)`` feeds one miscoverage indicator (or batch mean err rate
    in [0, 1]) and returns the *next* level alpha_{t+1}. ``alpha_history``
    records the level issued at each step (before the update), matching
    ``quant_fund.models.conformal.AdaptiveConformal``'s convention.
    """

    def __init__(
        self,
        alpha: float = 0.10,
        Ki: float | None = None,
        Kp: float | None = None,
        Kd: float = 0.0,
        clip: tuple[float, float] = (1e-4, 1.0 - 1e-4),
        quantile_levels: Array | None = None,
    ) -> None:
        self.alpha = _check_alpha(alpha)
        default_ki = 1.0 / float(np.sqrt(_REF_T))
        self.Ki = _check_gain("Ki", Ki, default_ki)
        self.Kp = _check_gain("Kp", Kp, self.Ki / 6.0)
        self.Kd = _check_gain("Kd", Kd, 0.0)
        self.clip = _check_clip(clip)
        self.quantile_levels = _check_levels(quantile_levels, self.alpha)
        self.alpha_t = self.alpha
        self._e_prev = 0.0
        self._e_prev2 = 0.0
        self._alpha_hist: list[float] = []
        self._cov_hist: list[float] = []

    def update(self, err: float) -> float:
        """Feed one miscoverage rate err_t in [0, 1]; return the next level."""
        r = float(err)
        if not np.isfinite(r) or not 0.0 <= r <= 1.0:
            raise ValueError("err must be a finite miscoverage rate in [0, 1]")
        e = self.alpha - r
        self._alpha_hist.append(self.alpha_t)
        self._cov_hist.append(1.0 - r)
        nxt = _pid_step(
            np.asarray([self.alpha_t], dtype=float),
            np.asarray([e], dtype=float),
            np.asarray([self._e_prev], dtype=float),
            np.asarray([self._e_prev2], dtype=float),
            ki=self.Ki,
            kp=self.Kp,
            kd=self.Kd,
            lo=self.clip[0],
            hi=self.clip[1],
        )
        self.alpha_t = float(nxt[0])
        self._e_prev2 = self._e_prev
        self._e_prev = e
        return self.alpha_t

    @property
    def alpha_history(self) -> Array:
        """Level issued at each step, shape (n_steps,)."""
        return np.asarray(self._alpha_hist, dtype=float)

    @property
    def coverage_history(self) -> Array:
        """Per-step coverage indicators 1 - err_t, shape (n_steps,)."""
        return np.asarray(self._cov_hist, dtype=float)

    @property
    def n_steps_(self) -> int:
        return len(self._alpha_hist)


class ConformalPIDQuantile:
    """PID-controlled conformalizer for a quantile-grid forecast.

    One scalar PID loop per quantile level tau_j (shared gains), each
    regulating the one-sided upper coverage P(y <= q̃_j) -> tau_j on the
    score store s_t = y_t - q_tau_j (the per-level convention of
    quant_fund.models.agaci). A two-sided central band is the pair of tail
    levels [alpha/2, 1 - alpha/2]; its miscoverage is tracked by those two
    loops with the total budget alpha split across the tails.

    Parameters
    ----------
    alpha:
        Target miscoverage in (0, 1); used to derive the default levels.
    Ki, Kp, Kd, clip:
        As in ``ConformalPID``; one recursion per level.
    quantile_levels:
        Strictly increasing levels in (0, 1). None -> the two-sided grid
        (alpha/2, 1 - alpha/2).

    Notes
    -----
    ``update(y, quantile_grid)`` returns the conformalized grid *before*
    incorporating ``y`` (the prediction issued at the current step), then
    updates every level loop and score store. ``alpha_history`` and
    ``coverage_history`` record, per step, the issued levels and the
    per-level one-sided coverage indicators 1{y <= q̃_j} of the grid.
    """

    def __init__(
        self,
        alpha: float = 0.10,
        Ki: float | None = None,
        Kp: float | None = None,
        Kd: float = 0.0,
        clip: tuple[float, float] = (1e-4, 1.0 - 1e-4),
        quantile_levels: Array | None = None,
    ) -> None:
        self.alpha = _check_alpha(alpha)
        default_ki = 1.0 / float(np.sqrt(_REF_T))
        self.Ki = _check_gain("Ki", Ki, default_ki)
        self.Kp = _check_gain("Kp", Kp, self.Ki / 6.0)
        self.Kd = _check_gain("Kd", Kd, 0.0)
        self.clip = _check_clip(clip)
        self._levels = _check_levels(quantile_levels, self.alpha)
        self._alpha_levels = 1.0 - self._levels
        self._alpha_t = self._alpha_levels.copy()
        self._e_prev = np.zeros(self._levels.size, dtype=float)
        self._e_prev2 = np.zeros(self._levels.size, dtype=float)
        self._scores: list[list[float]] = [[] for _ in range(self._levels.size)]
        self._n = 0
        self.alpha_history: list[Array] = []
        self.coverage_history: list[Array] = []

    @property
    def quantile_levels_(self) -> Array:
        return self._levels.copy()

    @property
    def alpha_t_(self) -> Array:
        """Current issued per-level miscoverage targets, shape (n_levels,)."""
        return self._alpha_t.copy()

    @property
    def n_steps_(self) -> int:
        return self._n

    def update(self, y: float, quantile_grid: Array) -> Array:
        """Issue the PID-conformalized non-crossing grid, then learn from ``y``.

        Parameters
        ----------
        y:
            Scalar realized value, finite.
        quantile_grid:
            1-d raw quantile forecasts, one per level, same length as
            ``quantile_levels``, all finite.

        Returns
        -------
        np.ndarray
            Non-crossing conformalized quantile grid for the current step.
        """
        y_arr = np.asarray(y, dtype=float)
        if y_arr.size != 1:
            raise ValueError("y must be a scalar")
        yv = float(y_arr.ravel()[0])
        if not np.isfinite(yv):
            raise ValueError("y must be finite")
        grid = np.asarray(quantile_grid, dtype=float)
        n_lvl = int(self._levels.size)
        if grid.ndim != 1 or grid.size != n_lvl:
            raise ValueError(f"grid must be 1-d with {n_lvl} entries (one per quantile level)")
        if not np.all(np.isfinite(grid)):
            raise ValueError("grid must contain only finite values")

        offsets = np.zeros(n_lvl, dtype=float)
        for j in range(n_lvl):
            hist = self._scores[j]
            if hist:
                offsets[j] = conformal_quantile(
                    np.asarray(hist, dtype=float), float(self._alpha_t[j])
                )
        q_tilde = grid + offsets
        q_out = np.maximum.accumulate(q_tilde)
        q_out = rearrange_quantiles(q_out.reshape(1, -1)).ravel()
        self.coverage_history.append((yv <= q_out).astype(float))
        self.alpha_history.append(self._alpha_t.copy())

        err = (yv > q_tilde).astype(float)
        e = self._alpha_levels - err
        self._alpha_t = _pid_step(
            self._alpha_t,
            e,
            self._e_prev,
            self._e_prev2,
            ki=self.Ki,
            kp=self.Kp,
            kd=self.Kd,
            lo=self.clip[0],
            hi=self.clip[1],
        )
        self._e_prev2 = self._e_prev
        self._e_prev = e
        for j in range(n_lvl):
            self._scores[j].append(yv - float(grid[j]))
        self._n += 1
        return q_out
