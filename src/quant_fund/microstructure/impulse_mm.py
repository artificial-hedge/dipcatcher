"""Hawkes-excited impulse market making (Guéant-style Riccati MM).

**Labeled SYNTHETIC** research infrastructure: a market maker whose optimal
bid/ask offsets are conditioned on a bivariate Hawkes market-order intensity
``(lambda^+_t, lambda^-_t)``, solved through the coupled Riccati-ODE /
HJB system in ``(t, q, lambda^+, lambda^-)`` with a finite-difference
backward shooting solver in pure numpy. Composes the lane B4-i ZI-LOB
engine (:mod:`quant_fund.microstructure.zi_lob_simulator`) without
modifying it: :class:`HawkesFlow` plugs into ``ZILobSimulator(flow=...)``
through the ``MarkovRegimeFlow`` contract, and
:func:`run_impulse_mm_session` is a ``run_mm_session``-style runner extended
with an online bivariate-Hawkes intensity filter.

References:

- Hawkes (1971). Spectra of some self-exciting and mutually exciting point
  processes. *Biometrika* 58 — exponential-kernel intensity.
- Gueant, Lehalle, Fernandez-Tapia (2012). Optimal portfolio liquidation
  with limit orders. *Operations Research* 60(3) — CARA ansatz reducing the
  stochastic control problem to a coupled Riccati system over inventory.
- Cartea, Jaimungal, Penalva (2015). *Algorithmic and High-Frequency
  Trading*, ch. 11 — market making with stochastic (incl. self-exciting)
  order-flow intensity: the ``(q, lambda)``-dependent HJB and the
  jump-shifted optimal-depth formula implemented here.
- Avellaneda, Stoikov (2008). *Quantitative Finance* 8(3) — the Poisson
  limit: with zero excitation the Riccati solution must recover the AS
  reservation-price skew ``-q gamma sigma^2 tau`` and half-spread
  ``gamma sigma^2 tau / 2 + (1/gamma) ln(1 + gamma/kappa)`` (asserted in
  tests).
- Bowsher (2007); Filimonov, Sornette (2012) — bivariate MO-flow Hawkes
  parameterization; kernel convention ``phi = alpha * beta * exp(-beta t)``
  shared with ``models.point_process``.

Model (formulation implemented here):

- Buy/sell market orders arrive as a bivariate Hawkes process,
  ``lambda^i(t) = mu_i + sum_j alpha_{ij} beta e^{-beta (t - t_j)}`` over
  past events ``j`` of both sides. ``alpha`` is the 2x2 branching matrix;
  stationarity requires spectral radius ``rho(alpha) < 1`` (fail-closed).
- Fill model: a posted ask at offset ``delta^a`` from mid is lifted by buy
  MOs with intensity ``lambda^+ * A * exp(-kappa * delta^a)`` (Gueant's
  exponential depth law); symmetrically for the bid. MOs that do not reach
  the quote still arrive and still excite ``lambda`` — the "miss" jump
  channel of the HJB.
- Value ansatz ``u = -exp(-gamma (x + q s + theta_q(t, lambda+, lambda-)))``
  turns the HJB into the coupled system (Cartea et al. 2015 ch. 11)::

      dt theta_q = (gamma sigma^2 / 2) q^2
                   - beta (mu+ - lambda+) d+ theta_q - beta (mu- - lambda-) d- theta_q
                   - lambda+ [ A e^{-kappa da*} R_a / (kappa + gamma) + (1 - R_a)/gamma ]
                   - lambda- [ A e^{-kappa db*} R_b / (kappa + gamma) + (1 - R_b)/gamma ]

  with post-jump evaluation
  ``R_a = exp(-gamma (theta_q^{post-buy} - theta_q))``,
  ``da* = (1/gamma) ln(1 + gamma/kappa) + theta_q^{post-buy} - theta_{q-1}^{post-buy}``
  (and the mirrored ``db*`` over the post-sell state). Terminal condition
  ``theta_q(T) = 0`` (liquidate at mid, the AS convention). At the
  inventory walls ``q = +/- q_max`` the maker posts only the reducing side:
  the blocked-side fill term drops and its offset is ``+inf``.

Deviations from the reference math (deliberate, documented):

1. **Numerical depth clip.** ``delta*`` is floored at ``-neg_clip/kappa``
   inside the solver only (``e^{-kappa delta}`` would otherwise overflow the
   explicit integrator in extreme-skew states). It binds only where the
   exponential-intensity model has already degenerated (a deeply crossed
   quote); the lookup layer still reports the unclipped value.
2. **Event-time HawkesFlow.** The ``RegimeState`` contract
   (``current()->(name, intensity_mult, p_buy)``; ``advance()`` per MO) is
   side- and time-blind — ``ZILobSimulator`` passes neither. ``HawkesFlow``
   therefore implements the *pooled* Hawkes on the MO clock (geometric
   kernel in event index), which is exactly the Hawkes family the contract
   can express; the continuous-time bivariate recursion lives in
   :class:`BivariateHawkes` (used by the solver and by the session runner's
   estimator, which sees real trade times and sides).
3. **Maker knows the flow model.** The session filter is perfectly
   specified (solver parameters are the estimator's); parameter-estimation
   error is out of scope for a correctness module.

Honesty: every output is a SYNTHETIC correctness diagnostic on a synthetic
engine, never market evidence. All P&L-like keys are namespaced
``sim_internal_*`` and must never be headlined; no broker connectivity or
live-trading claim anywhere. Pure numpy — no torch anywhere in this module.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from quant_fund.microstructure.zi_lob_simulator import (
    MM_TAG,
    ZI_LOB_REVISION,
    MarkovRegimeFlow,
    RegimeState,
    TradeEvent,
    ZILobConfig,
    ZILobSimulator,
    book_phase_metrics,
    regime_flow_diagnostics,
)
from quant_fund.utils.receipt import seal_receipt
from quant_fund.utils.reproducibility import git_revision

Array = NDArray[np.float64]

__all__ = [
    "HAWKES_MM_REVISION",
    "BivariateHawkes",
    "BivariateHawkesParams",
    "HawkesFlow",
    "ImpulseMMConfig",
    "ImpulseMMState",
    "ImpulseMMSolution",
    "ImpulseQuotePolicy",
    "hawkes_intensity_path",
    "hawkes_mm_bench",
    "hawkes_mm_policy",
    "run_impulse_mm_session",
    "solve_impulse_mm",
]

HAWKES_MM_REVISION = "SYNTHETIC_HAWKES_MM_v1"


# ---------------------------------------------------------------------------
# Fail-closed validation helpers (local; zi_lob_simulator's are private)
# ---------------------------------------------------------------------------


def _pos_finite(x: float, name: str) -> float:
    v = float(x)
    if not math.isfinite(v) or v <= 0.0:
        raise ValueError(f"{name} must be positive and finite, got {x!r}")
    return v


def _nonneg_finite(x: float, name: str) -> float:
    v = float(x)
    if not math.isfinite(v) or v < 0.0:
        raise ValueError(f"{name} must be non-negative and finite, got {x!r}")
    return v


def _prob(x: float, name: str) -> float:
    v = float(x)
    if not math.isfinite(v) or v < 0.0 or v > 1.0:
        raise ValueError(f"{name} must be a probability in [0, 1], got {x!r}")
    return v


def _int_at_least(x: int, floor: int, name: str) -> int:
    if isinstance(x, bool) or not isinstance(x, int) or x < floor:
        raise ValueError(f"{name} must be an int >= {floor}, got {x!r}")
    return x


def _check_side(side: str) -> str:
    if side not in ("buy", "sell"):
        raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")
    return side


# ---------------------------------------------------------------------------
# Bivariate Hawkes intensity (continuous-time, O(1) recursive form)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BivariateHawkesParams:
    """Bivariate exponential-kernel Hawkes parameters for the MO stream.

    ``alpha_*`` entries are dimensionless branching ratios:
    ``alpha_pp``/``alpha_mm`` self-excitation of buy/sell MOs,
    ``alpha_pm``/``alpha_mp`` cross-excitation (first index is the excited
    process: ``alpha_mp`` is the buy-MO -> sell-intensity channel). Each
    event of type ``j`` instantaneously adds ``alpha_{ij} * beta`` to
    ``lambda_i``; the contribution then decays at rate ``beta`` (shared
    decay, matching ``models.point_process.hawkes2_mle``). Stationarity
    requires spectral radius ``rho(alpha) < 1`` (fail-closed).
    """

    mu_plus: float
    mu_minus: float
    alpha_pp: float
    alpha_pm: float
    alpha_mp: float
    alpha_mm: float
    beta: float

    def __post_init__(self) -> None:
        _pos_finite(self.mu_plus, "mu_plus")
        _pos_finite(self.mu_minus, "mu_minus")
        for name in ("alpha_pp", "alpha_pm", "alpha_mp", "alpha_mm"):
            _nonneg_finite(getattr(self, name), name)
        _pos_finite(self.beta, "beta")
        if self.branching_ratio >= 1.0:
            raise ValueError(
                "explosive Hawkes excitation: spectral radius of the branching "
                f"matrix must be < 1, got {self.branching_ratio!r}"
            )

    @property
    def branching_matrix(self) -> Array:
        return np.array(
            [[self.alpha_pp, self.alpha_pm], [self.alpha_mp, self.alpha_mm]],
            dtype=np.float64,
        )

    @property
    def branching_ratio(self) -> float:
        eigs = np.linalg.eigvals(self.branching_matrix)
        return float(np.max(np.abs(eigs)))

    @property
    def stationary_mean(self) -> tuple[float, float]:
        """E[lambda] = (I - alpha)^{-1} mu under stationarity."""
        mat = np.eye(2) - self.branching_matrix
        mean = np.linalg.solve(mat, np.array([self.mu_plus, self.mu_minus]))
        return (float(mean[0]), float(mean[1]))

    @property
    def jump_buy(self) -> tuple[float, float]:
        """Instantaneous (d lambda+, d lambda-) added by a buy-MO event."""
        return (self.alpha_pp * self.beta, self.alpha_mp * self.beta)

    @property
    def jump_sell(self) -> tuple[float, float]:
        """Instantaneous (d lambda+, d lambda-) added by a sell-MO event."""
        return (self.alpha_pm * self.beta, self.alpha_mm * self.beta)


class BivariateHawkes:
    """Continuous-time bivariate Hawkes intensity via the O(1) recursion.

    Between events the intensities decay toward baseline:
    ``lambda_i(t) = mu_i + (lambda_i(t_last) - mu_i) e^{-beta (t - t_last)}``.
    ``observe(t, side)`` decays to ``t``, reports the pre-jump intensity
    (the intensity that "caused" the event under thinning), then applies the
    event's jump — the classic recursive evaluation, O(1) per event.
    Deterministic; fail-closed on backward or non-finite times.
    """

    def __init__(
        self,
        params: BivariateHawkesParams,
        *,
        lam0: tuple[float, float] | None = None,
    ) -> None:
        if not isinstance(params, BivariateHawkesParams):
            raise TypeError("params must be a BivariateHawkesParams")
        self._p = params
        if lam0 is None:
            lam0 = (params.mu_plus, params.mu_minus)
        lp, lm = float(lam0[0]), float(lam0[1])
        if not (math.isfinite(lp) and math.isfinite(lm)) or lp <= 0.0 or lm <= 0.0:
            raise ValueError(f"lam0 must be positive and finite, got {lam0!r}")
        self._lp = lp
        self._lm = lm
        self._t = 0.0
        self.n_events = 0

    @property
    def params(self) -> BivariateHawkesParams:
        return self._p

    @property
    def t(self) -> float:
        return self._t

    @property
    def lam_plus(self) -> float:
        return self._lp

    @property
    def lam_minus(self) -> float:
        return self._lm

    def intensity(self) -> tuple[float, float]:
        return (self._lp, self._lm)

    def advance_to(self, t: float) -> tuple[float, float]:
        """Decay both intensities to time ``t``; returns ``(lambda+, lambda-)``."""
        tt = float(t)
        if not math.isfinite(tt):
            raise ValueError(f"t must be finite, got {t!r}")
        if tt < self._t - 1e-12:
            raise ValueError(f"cannot decay intensity backward: t={tt} < last={self._t}")
        if tt > self._t:
            decay = math.exp(-self._p.beta * (tt - self._t))
            self._lp = self._p.mu_plus + (self._lp - self._p.mu_plus) * decay
            self._lm = self._p.mu_minus + (self._lm - self._p.mu_minus) * decay
            self._t = tt
        return self.intensity()

    def observe(self, t: float, side: str) -> tuple[float, float]:
        """Absorb one MO event: decay to ``t``, then apply the side's jump.

        Returns the pre-jump intensity ``(lambda+, lambda-)`` — the value an
        Ogata-thinning sampler or the Hawkes likelihood would use.
        """
        s = _check_side(side)
        out = self.advance_to(t)
        jp, jm = self._p.jump_buy if s == "buy" else self._p.jump_sell
        self._lp += jp
        self._lm += jm
        self.n_events += 1
        return out


def hawkes_intensity_path(
    times: Array,
    buys: Array,
    sells: Array,
    params: BivariateHawkesParams,
    *,
    kernel_span: float = 12.0,
) -> dict[str, Array]:
    """Intensity path on a uniform grid via discrete kernel convolution.

    Bins each side's event times onto ``times`` (uniform, strictly
    increasing), then convolves the binned counts with the truncated
    exponential kernel ``beta e^{-beta k dt}`` (``kernel_span`` decay
    half-lives of support, mass ``1 - e^{-kernel_span}``). Per-side::

        lambda+ = mu+ + alpha_pp * S_buy + alpha_pm * S_sell
        lambda- = mu- + alpha_mp * S_buy + alpha_mm * S_sell

    where ``S_*`` is the decay-sum convolution. Binning quantizes each event
    to its nearest grid point (error <= dt/2 in the kernel argument);
    ``times`` must be a uniform grid (fail-closed otherwise).
    """
    if not isinstance(params, BivariateHawkesParams):
        raise TypeError("params must be a BivariateHawkesParams")
    t = np.asarray(times, dtype=np.float64).ravel()
    if t.size < 2 or not np.all(np.isfinite(t)):
        raise ValueError("times must contain >= 2 finite points")
    dt = float(np.diff(t).mean())
    if dt <= 0.0 or not np.all(np.abs(np.diff(t) - dt) <= 1e-9 * max(1.0, dt)):
        raise ValueError("times must be a uniform strictly-increasing grid")
    kb = _pos_finite(kernel_span, "kernel_span")

    def _decay_sum(events: Array) -> Array:
        e = np.asarray(events, dtype=np.float64).ravel()
        if not np.all(np.isfinite(e)):
            raise ValueError("event times must be finite")
        counts = np.zeros(t.size)
        in_range = e[(e >= t[0] - 0.5 * dt) & (e <= t[-1] + 0.5 * dt)]
        if in_range.size:
            bins = np.clip(((in_range - t[0]) / dt).round().astype(np.int64), 0, t.size - 1)
            counts += np.bincount(bins, minlength=t.size)
        n_ker = int(math.ceil(kb / params.beta / dt)) + 1
        ker = params.beta * np.exp(-params.beta * dt * np.arange(n_ker))
        return np.convolve(counts, ker)[: t.size]

    sb = _decay_sum(np.asarray(buys, dtype=np.float64))
    ss = _decay_sum(np.asarray(sells, dtype=np.float64))
    return {
        "times": t,
        "lam_plus": params.mu_plus + params.alpha_pp * sb + params.alpha_pm * ss,
        "lam_minus": params.mu_minus + params.alpha_mp * sb + params.alpha_mm * ss,
    }


# ---------------------------------------------------------------------------
# HawkesFlow: pooled event-time Hawkes under the RegimeState contract
# ---------------------------------------------------------------------------


class HawkesFlow:
    """Pooled self-exciting MO-flow modulator (``RegimeState`` contract).

    The ZI-LOB flow contract is event-time only — ``advance()`` carries no
    event time and no side — so this is the *event-time* Hawkes: every
    market order kicks the pooled normalized intensity by ``excitation``
    and it decays geometrically by ``decay_rho`` per MO::

        I_n = 1 + sum_{j <= n} excitation * decay_rho ** (n - j)
        intensity_mult = I_n      (bound: 1 + excitation / (1 - decay_rho))

    ``p_buy`` is a static directional split. Deterministic (no RNG).
    ``current()`` yields the same ``RegimeState`` triple
    ``MarkovRegimeFlow`` produces, so ``ZILobSimulator(flow=HawkesFlow)``,
    ``run_mm_session(flow=...)`` and :func:`run_impulse_mm_session` consume
    it unchanged. See the module docstring for why the continuous-time
    bivariate excitation lives in :class:`BivariateHawkes` instead.
    """

    def __init__(
        self,
        *,
        excitation: float,
        decay_rho: float,
        p_buy: float = 0.5,
        name: str = "hawkes_excited",
    ) -> None:
        self._eps = _nonneg_finite(excitation, "excitation")
        rho = float(decay_rho)
        if not math.isfinite(rho) or rho <= 0.0 or rho >= 1.0:
            raise ValueError(f"decay_rho must lie in (0, 1), got {decay_rho!r}")
        self._rho = rho
        self._p_buy = _prob(p_buy, "p_buy")
        if not isinstance(name, str) or not name:
            raise ValueError("name must be a non-empty string")
        self._name = name
        self._intensity = 1.0
        self.n_mo = 0

    @property
    def intensity_mult_bound(self) -> float:
        """Supremum of ``intensity_mult``: 1 + eps / (1 - rho)."""
        return 1.0 + self._eps / (1.0 - self._rho)

    def current(self) -> RegimeState:
        return RegimeState(self._name, self._intensity, self._p_buy)

    def advance(self) -> None:
        """One MO-clock event: geometric decay, then the excitation kick."""
        self._intensity = 1.0 + (self._intensity - 1.0) * self._rho + self._eps
        self.n_mo += 1

    def expected_p_buy(self) -> float:
        """Mean buy probability realized so far (static here)."""
        if self.n_mo == 0:
            raise ValueError("expected_p_buy undefined before any MO event")
        return self._p_buy


# ---------------------------------------------------------------------------
# Riccati solver: theta_q(t, lambda+, lambda-) and optimal offsets
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ImpulseMMConfig:
    """Solver configuration for the Hawkes-excited impulse market maker.

    ``gamma`` CARA risk aversion, ``sigma`` mid diffusion (price/sqrt(time)),
    ``kappa`` exponential fill-intensity decay in inverse price units,
    ``a_fill`` the Guéant depth-law scale (fill intensity
    ``lambda * a_fill * exp(-kappa delta)``; must be <= 1 so the fill
    intensity stays a fraction of the arrival rate), ``q_max`` the hard
    inventory bound. ``hawkes`` carries the bivariate flow parameters.

    Grids: uniform lambda grids ``[0, lam_max]`` x ``[0, lam_max]`` with
    ``n_lam`` nodes per side (must bracket the stationary mean — post-jump
    states above ``lam_max`` are clamped to the edge, a documented boundary
    approximation), and ``n_t`` output steps over horizon ``horizon``.
    Internal substeps are chosen for the explicit-Euler CFL (drift and fill
    terms); ``n_sub`` overrides the auto choice.
    """

    gamma: float
    sigma: float
    kappa: float
    a_fill: float
    q_max: int
    hawkes: BivariateHawkesParams
    horizon: float
    n_t: int = 200
    n_lam: int = 25
    lam_max: float = 0.0
    n_sub: int = 0
    neg_clip: float = 10.0

    def __post_init__(self) -> None:
        _pos_finite(self.gamma, "gamma")
        _pos_finite(self.sigma, "sigma")
        _pos_finite(self.kappa, "kappa")
        a = _pos_finite(self.a_fill, "a_fill")
        if a > 1.0:
            raise ValueError(f"a_fill must lie in (0, 1] (fill fraction), got {self.a_fill!r}")
        _int_at_least(self.q_max, 1, "q_max")
        if not isinstance(self.hawkes, BivariateHawkesParams):
            raise TypeError("hawkes must be a BivariateHawkesParams")
        _pos_finite(self.horizon, "horizon")
        _int_at_least(self.n_t, 8, "n_t")
        _int_at_least(self.n_lam, 5, "n_lam")
        lam_max = _pos_finite(self.lam_max, "lam_max")
        mu_hi = max(self.hawkes.mu_plus, self.hawkes.mu_minus)
        if lam_max <= mu_hi:
            raise ValueError(
                f"lam_max must exceed the Hawkes baselines ({mu_hi!r}), got {self.lam_max!r}"
            )
        mp, mm = self.hawkes.stationary_mean
        if lam_max <= mp or lam_max <= mm:
            raise ValueError(
                "lam_max must exceed the stationary mean intensities "
                f"({mp!r}, {mm!r}), got {self.lam_max!r}"
            )
        if self.n_sub != 0:
            _int_at_least(self.n_sub, 1, "n_sub")
        _pos_finite(self.neg_clip, "neg_clip")


ShiftWeights = tuple[
    NDArray[np.int64], NDArray[np.int64], Array, Array
]  # (i0, i1, w0, w1) linear-interpolation slots


def _shift_weights(grid: Array, shift: float) -> ShiftWeights:
    """Interpolation indices/weights for evaluating a grid function at x+shift.

    Post-jump intensities are evaluated by linear interpolation on the
    lambda grid; coordinates above ``grid[-1]`` clamp to the edge (the
    documented ``lam_max`` boundary approximation). ``shift >= 0``.
    """
    x = np.clip(grid + shift, grid[0], grid[-1])
    i0 = np.clip(np.searchsorted(grid, x, side="right") - 1, 0, grid.size - 2).astype(np.int64)
    lo, hi = grid[i0], grid[i0 + 1]
    w1 = np.where(hi > lo, (x - lo) / np.where(hi > lo, hi - lo, 1.0), 0.0)
    return i0, i0 + 1, 1.0 - w1, w1


def _shift_eval(theta: Array, wp: ShiftWeights, wm: ShiftWeights) -> Array:
    """Bilinear interpolation of ``theta`` at (lp + dlp, lm + dlm) for all nodes."""
    i0, i1, w0, w1 = wp
    j0, j1, v0, v1 = wm
    t1 = w0[:, None] * theta[i0, :] + w1[:, None] * theta[i1, :]
    return v0[None, :] * t1[:, j0] + v1[None, :] * t1[:, j1]


def _upwind(theta: Array, vel: Array, d: float, axis: int) -> Array:
    """First-order upwind difference of ``theta`` along ``axis``.

    ``vel`` is the advection velocity per node along that axis (drift toward
    the Hawkes baseline). The one-sided difference reads toward the
    upwind neighbor — forward where ``vel > 0``, backward where ``vel < 0``.
    Edge nodes always face the inward direction (``lam_max > mu`` is
    validated), so the wrapped ``np.roll`` values are never selected.
    """
    fwd = (np.roll(theta, -1, axis=axis) - theta) / d
    bwd = (theta - np.roll(theta, 1, axis=axis)) / d
    shape = [1] * theta.ndim
    shape[axis] = vel.size
    return np.where(vel.reshape(shape) > 0.0, fwd, bwd)


@dataclass(frozen=True)
class ImpulseMMSolution:
    """Backward-solved Riccati system and the optimal-offset grids.

    ``theta[it, iq, ilp, ilm]`` is the value correction; ``delta_ask`` /
    ``delta_bid`` are the optimal mid-relative offsets on the same grid
    (``+inf`` where the inventory wall suppresses the quote). ``t_grid``
    counts down to the terminal time (index ``it`` corresponds to solver
    time ``t = it * dt``; ``tau`` remaining time is ``horizon - t``).
    """

    config: ImpulseMMConfig
    t_grid: Array
    q_grid: Array
    lam_grid: Array
    theta: Array
    delta_ask: Array
    delta_bid: Array
    n_inner_steps: int

    def offsets(
        self,
        tau: float,
        q: int,
        lam_plus: float,
        lam_minus: float,
    ) -> tuple[float, float]:
        """Optimal ``(bid_offset, ask_offset)`` at remaining time ``tau``.

        Bilinear in ``(lambda+, lambda-)``, linear in time. ``q`` must be an
        int inside ``[-q_max, q_max]``; at the walls the suppressed side
        returns ``+inf`` (the policy layer turns that into "no quote").
        """
        tt = float(tau)
        if not math.isfinite(tt) or tt < 0.0:
            raise ValueError(f"tau must be non-negative and finite, got {tau!r}")
        if isinstance(q, bool) or not isinstance(q, int):
            raise ValueError(f"q must be an int, got {q!r}")
        if abs(q) > self.config.q_max:
            raise ValueError(f"|q| must be <= q_max, got {q!r}")
        lp = float(lam_plus)
        lm = float(lam_minus)
        for name, v in (("lam_plus", lp), ("lam_minus", lm)):
            if not math.isfinite(v) or v < 0.0:
                raise ValueError(f"{name} must be non-negative and finite, got {v!r}")
        cfg = self.config
        t_solve = min(max(cfg.horizon - tt, 0.0), cfg.horizon)
        dt = float(self.t_grid[1] - self.t_grid[0]) if self.t_grid.size > 1 else 1.0
        it_f = t_solve / dt
        it0 = int(min(max(math.floor(it_f), 0), self.t_grid.size - 2))
        wt = min(max(it_f - it0, 0.0), 1.0)
        iq = q + cfg.q_max
        db = _bilinear_lookup(self.delta_bid, it0, wt, iq, self.lam_grid, lp, lm)
        da = _bilinear_lookup(self.delta_ask, it0, wt, iq, self.lam_grid, lp, lm)
        return (db, da)


def _point_weights(grid: Array, x: float) -> tuple[int, int, float, float]:
    """Linear-interpolation (i0, i1, w0, w1) for a scalar point on ``grid``."""
    xx = min(max(x, float(grid[0])), float(grid[-1]))
    i0 = int(np.clip(np.searchsorted(grid, xx, side="right") - 1, 0, grid.size - 2))
    lo, hi = float(grid[i0]), float(grid[i0 + 1])
    w1 = (xx - lo) / (hi - lo) if hi > lo else 0.0
    return i0, i0 + 1, 1.0 - w1, w1


def _bilinear_lookup(
    grid4: Array,
    it0: int,
    wt: float,
    iq: int,
    lam_grid: Array,
    lp: float,
    lm: float,
) -> float:
    """Interpolate ``grid4[it, iq, ilp, ilm]`` at (t-blend, iq, lp, lm)."""
    i0, i1, w0, w1 = _point_weights(lam_grid, lp)
    j0, j1, v0, v1 = _point_weights(lam_grid, lm)

    def at(it: int) -> float:
        g = grid4[it, iq]
        vals = np.array([g[i0, j0], g[i0, j1], g[i1, j0], g[i1, j1]])
        if not np.all(np.isfinite(vals)):
            # Suppressed (inventory-wall) rows are uniformly +inf; a mixed
            # corner set cannot arise structurally, so any inf means the
            # quote is suppressed at this point.
            return float("inf")
        return float(w0 * (v0 * vals[0] + v1 * vals[1]) + w1 * (v0 * vals[2] + v1 * vals[3]))

    if wt <= 0.0:
        return at(it0)
    if wt >= 1.0:
        return at(it0 + 1)
    a0, a1 = at(it0), at(it0 + 1)
    if a0 == float("inf") or a1 == float("inf"):
        return float("inf")
    return (1.0 - wt) * a0 + wt * a1


def solve_impulse_mm(config: ImpulseMMConfig) -> ImpulseMMSolution:
    """Solve the coupled Riccati-ODE system by backward shooting.

    Explicit backward Euler on the output grid with ``n_sub`` auto-chosen
    inner steps (CFL heuristic on the drift and fill terms), first-order
    upwind differences for the Hawkes mean-reversion advection, and bilinear
    grid interpolation for the post-jump value evaluation. Deterministic —
    pure numpy arithmetic, no RNG. Fail-closed: non-finite ``theta`` at any
    stored level raises ``RuntimeError`` (the honesty contract prefers a
    loud failure over a silently wrong receipt).
    """
    if not isinstance(config, ImpulseMMConfig):
        raise TypeError("config must be an ImpulseMMConfig")
    cfg = config
    p = cfg.hawkes
    n_t, n_lam, nq = cfg.n_t, cfg.n_lam, 2 * cfg.q_max + 1
    q_grid = np.arange(-cfg.q_max, cfg.q_max + 1, dtype=np.float64)
    lam_grid = np.linspace(0.0, cfg.lam_max, n_lam)
    t_grid = np.linspace(0.0, cfg.horizon, n_t + 1)
    dt_out = cfg.horizon / n_t
    dlam = float(lam_grid[1] - lam_grid[0])

    c0 = math.log(1.0 + cfg.gamma / cfg.kappa) / cfg.gamma
    d_min = -cfg.neg_clip / cfg.kappa
    jbp, jbm = p.jump_buy  # buy-MO (fills the ask) post-jump intensity shift
    jsp, jsm = p.jump_sell  # sell-MO (fills the bid) post-jump shift
    w_buy = (_shift_weights(lam_grid, jbp), _shift_weights(lam_grid, jbm))
    w_sell = (_shift_weights(lam_grid, jsp), _shift_weights(lam_grid, jsm))

    vel_p = p.beta * (p.mu_plus - lam_grid)
    vel_m = p.beta * (p.mu_minus - lam_grid)
    q_cost = 0.5 * cfg.gamma * cfg.sigma * cfg.sigma * q_grid * q_grid

    if cfg.n_sub > 0:
        n_sub = cfg.n_sub
    else:
        # CFL heuristic: drift advection needs dt*|vel|/dlam <= ~0.5; the fill
        # channel's effective rate is lambda * a_fill * kappa / (kappa+gamma)
        # in the unclipped regime (the neg_clip rail is a safety bound, not
        # the operating point). Deterministic; fine grids may still stiffen —
        # non-finite theta then fails closed.
        vel_max = float(np.max(np.abs(vel_p)) + np.max(np.abs(vel_m)))
        dt_drift = 0.5 * dlam / max(vel_max, 1e-12)
        rate_fill = (
            cfg.lam_max * cfg.a_fill * cfg.kappa / (cfg.kappa + cfg.gamma) + cfg.lam_max / cfg.gamma
        )
        dt_fill = 0.5 / max(rate_fill, 1e-12)
        n_sub = int(max(1, min(64, math.ceil(dt_out / min(dt_drift, dt_fill)))))
    dt = dt_out / n_sub

    theta = np.zeros((n_t + 1, nq, n_lam, n_lam), dtype=np.float64)
    delta_ask = np.full((n_t + 1, nq, n_lam, n_lam), np.inf, dtype=np.float64)
    delta_bid = np.full((n_t + 1, nq, n_lam, n_lam), np.inf, dtype=np.float64)
    lam_p2 = lam_grid[:, None]
    lam_m2 = lam_grid[None, :]

    def _rhs(th: Array) -> Array:
        """dtheta/dtau (backward-time RHS); ``th`` is (nq, nlp, nlm)."""
        # Post-jump value functions (bilinear-shifted by each side's jump).
        tb = np.stack([_shift_eval(th[k], w_buy[0], w_buy[1]) for k in range(nq)])
        ts = np.stack([_shift_eval(th[k], w_sell[0], w_sell[1]) for k in range(nq)])
        r_a = np.exp(-cfg.gamma * (tb - th))
        r_b = np.exp(-cfg.gamma * (ts - th))
        # Fill + miss channels; suppressed fill at the inventory walls.
        ask_gain = np.zeros_like(th)
        bid_gain = np.zeros_like(th)
        ask_gain[1:] = (
            lam_p2
            * cfg.a_fill
            / (cfg.kappa + cfg.gamma)
            * np.exp(-cfg.kappa * np.maximum(c0 + tb[1:] - tb[:-1], d_min))
            * r_a[1:]
        )
        bid_gain[:-1] = (
            lam_m2
            * cfg.a_fill
            / (cfg.kappa + cfg.gamma)
            * np.exp(-cfg.kappa * np.maximum(c0 + ts[:-1] - ts[1:], d_min))
            * r_b[:-1]
        )
        miss = lam_p2 * (1.0 - r_a) / cfg.gamma + lam_m2 * (1.0 - r_b) / cfg.gamma
        drift = np.zeros_like(th)
        for k in range(nq):
            drift[k] = vel_p[:, None] * _upwind(th[k], vel_p, dlam, axis=0) + vel_m[
                None, :
            ] * _upwind(th[k], vel_m, dlam, axis=1)
        return -q_cost[:, None, None] + drift + ask_gain + bid_gain + miss

    def _level_offsets(th: Array, it: int) -> None:
        tb = np.stack([_shift_eval(th[k], w_buy[0], w_buy[1]) for k in range(nq)])
        ts = np.stack([_shift_eval(th[k], w_sell[0], w_sell[1]) for k in range(nq)])
        delta_ask[it, 1:] = c0 + tb[1:] - tb[:-1]
        delta_bid[it, :-1] = c0 + ts[:-1] - ts[1:]

    _level_offsets(theta[n_t], n_t)
    n_inner = 0
    for it in range(n_t, 0, -1):
        th = theta[it]
        for _ in range(n_sub):
            th = th + dt * _rhs(th)
            n_inner += 1
        if not np.all(np.isfinite(th)):
            raise RuntimeError(
                f"Riccati solver diverged at t={t_grid[it - 1]:.6g} "
                "(non-finite theta — refine the grid or shrink the horizon)"
            )
        theta[it - 1] = th
        _level_offsets(th, it - 1)

    return ImpulseMMSolution(
        config=cfg,
        t_grid=t_grid,
        q_grid=q_grid,
        lam_grid=lam_grid,
        theta=theta,
        delta_ask=delta_ask,
        delta_bid=delta_bid,
        n_inner_steps=n_inner,
    )


# ---------------------------------------------------------------------------
# Policy + session runner
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ImpulseMMState:
    """Decision state handed to a Hawkes-conditioned quoting policy."""

    t: float
    mid: float | None
    best_bid: float | None
    best_ask: float | None
    inventory: int
    tau: float
    lam_plus: float
    lam_minus: float


ImpulseQuotePolicy = Callable[[ImpulseMMState], tuple[float | None, float | None]]


def hawkes_mm_policy(
    solution: ImpulseMMSolution,
    *,
    tick: float | None = None,
) -> ImpulseQuotePolicy:
    """ImpulseQuotePolicy adapter for :func:`solve_impulse_mm` output.

    Looks up the optimal offsets at ``(tau, inventory, lambda+, lambda-)``;
    an ``inf`` offset (inventory wall) suppresses that side. Optional tick
    snapping floors the bid and ceils the ask (same convention as the AS /
    GLFT adapters). Fail-closed on a malformed solution.
    """
    if not isinstance(solution, ImpulseMMSolution):
        raise TypeError("solution must be an ImpulseMMSolution")
    tk = _pos_finite(tick, "tick") if tick is not None else None

    def policy(state: ImpulseMMState) -> tuple[float | None, float | None]:
        if state.mid is None or state.tau <= 0.0:
            return (None, None)
        db, da = solution.offsets(
            min(state.tau, solution.config.horizon),
            int(state.inventory),
            state.lam_plus,
            state.lam_minus,
        )
        bid = None if not math.isfinite(db) else state.mid - db
        ask = None if not math.isfinite(da) else state.mid + da
        if tk is not None:
            if bid is not None:
                bid = math.floor(bid / tk + 1e-9) * tk
                if bid <= 0.0:
                    bid = None
            if ask is not None:
                ask = math.ceil(ask / tk - 1e-9) * tk
        if bid is not None and ask is not None and bid >= ask:
            return (None, None)
        return (bid, ask)

    return policy


def run_impulse_mm_session(
    *,
    config: ZILobConfig,
    policy: ImpulseQuotePolicy | None = None,
    solution: ImpulseMMSolution | None = None,
    horizon: float,
    decision_interval: float = 1.0,
    flow: HawkesFlow | MarkovRegimeFlow | None = None,
    inventory_cap: int | None = None,
    sample_interval: float = 25.0,
    estimator: BivariateHawkes | None = None,
) -> dict[str, Any]:
    """Run a Hawkes-conditioned market-making session in the ZI-LOB.

    Mirrors :func:`quant_fund.microstructure.zi_lob_simulator.run_mm_session`
    (same accounting, clipping and honesty schema) with two Hawkes
    additions: (a) the environment may be excited by a :class:`HawkesFlow`
    (or any ``RegimeState``-contract flow), and (b) the policy is fed an
    online :class:`BivariateHawkes` intensity estimate maintained from the
    *observable* MO tape — every printed trade's aggressor and time. MOs
    that hit an empty book side leave no print and are invisible to the
    maker, matching the public information set.

    Returns a **SYNTHETIC** diagnostic bundle; ``sim_internal_*`` keys are
    simulator-internal mark-to-market accounting, never headline metrics.
    """
    if not isinstance(config, ZILobConfig):
        raise TypeError("config must be a ZILobConfig")
    if policy is None:
        if solution is None:
            raise ValueError("either policy or solution must be provided")
        policy = hawkes_mm_policy(solution, tick=config.tick)
    if not callable(policy):
        raise TypeError("policy must be callable")
    h = _pos_finite(horizon, "horizon")
    di = _pos_finite(decision_interval, "decision_interval")
    si = _pos_finite(sample_interval, "sample_interval")
    cap: int | None = None
    if inventory_cap is not None:
        cap = _int_at_least(int(inventory_cap), 1, "inventory_cap")
    if estimator is not None and not isinstance(estimator, BivariateHawkes):
        raise TypeError("estimator must be a BivariateHawkes")

    # HawkesFlow satisfies the RegimeState contract structurally
    # (current()->RegimeState, advance() per MO); ZILobSimulator's parameter
    # annotation is nominal, so the duck-typed adapter is cast at the
    # boundary (zi_lob_simulator is not modified by this lane).
    sim = ZILobSimulator(config, flow=cast(MarkovRegimeFlow | None, flow))
    est = estimator
    inventory = 0
    cash = 0.0
    bid_oid: int | None = None
    ask_oid: int | None = None
    n_fills = n_fills_bid = n_fills_ask = 0
    n_cancels = n_clips = n_skipped = n_decisions = 0
    queue_ahead_fills: list[int] = []
    fill_waits: list[float] = []
    inv_path: list[int] = []
    inv_times: list[float] = []
    lam_p_path: list[float] = []
    lam_m_path: list[float] = []
    mtm_path: list[float] = []
    max_abs_inv = 0
    last_mid: float | None = sim.mid
    samples = [sim.sample()]
    signs: list[float] = []
    trade_cursor = 0

    def _cancel_outstanding() -> None:
        nonlocal bid_oid, ask_oid, n_cancels
        for oid in (bid_oid, ask_oid):
            if oid is not None and sim.cancel_order(oid):
                n_cancels += 1
        bid_oid = None
        ask_oid = None

    def _requote() -> None:
        nonlocal bid_oid, ask_oid, n_clips, n_skipped, n_decisions
        n_decisions += 1
        _cancel_outstanding()
        mid = sim.mid
        if mid is None:
            n_skipped += 1
            return
        if est is not None:
            lp, lm = est.advance_to(sim.t)
        else:
            lp, lm = 0.0, 0.0
        lam_p_path.append(lp)
        lam_m_path.append(lm)
        state = ImpulseMMState(
            t=sim.t,
            mid=mid,
            best_bid=sim.best_bid,
            best_ask=sim.best_ask,
            inventory=inventory,
            tau=h - sim.t,
            lam_plus=lp,
            lam_minus=lm,
        )
        bid_px, ask_px = policy(state)
        if cap is not None:
            if inventory >= cap:
                bid_px = None
            if inventory <= -cap:
                ask_px = None
        tick = config.tick
        ba, bb = sim.best_ask, sim.best_bid
        if bid_px is not None and ba is not None and bid_px >= ba - 1e-12:
            bid_px = ba - tick
            n_clips += 1
            if bid_px <= 0.0:
                bid_px = None
        if ask_px is not None and bb is not None and ask_px <= bb + 1e-12:
            ask_px = bb + tick
            n_clips += 1
        if bid_px is not None and ask_px is not None and bid_px >= ask_px:
            n_skipped += 1
            return
        if bid_px is not None:
            bid_oid = sim.submit_limit_order("buy", bid_px, MM_TAG)
        if ask_px is not None:
            ask_oid = sim.submit_limit_order("sell", ask_px, MM_TAG)

    def _record_path() -> None:
        nonlocal last_mid
        mid = sim.mid
        if mid is not None:
            last_mid = mid
        inv_path.append(inventory)
        inv_times.append(sim.t)
        if mid is not None:
            mtm_path.append(cash + inventory * mid)
        else:
            mtm_path.append(float("nan"))

    def _drain_trades() -> None:
        nonlocal trade_cursor, inventory, cash, bid_oid, ask_oid
        nonlocal n_fills, n_fills_bid, n_fills_ask, max_abs_inv
        while trade_cursor < len(sim.trades):
            tr: TradeEvent = sim.trades[trade_cursor]
            trade_cursor += 1
            signs.append(1.0 if tr.aggressor == "buy" else -1.0)
            if est is not None:
                est.observe(tr.t, tr.aggressor)
            if tr.maker_tag != MM_TAG:
                continue
            if tr.maker_side == "buy":
                inventory += tr.qty
                cash -= tr.price * tr.qty
                n_fills_bid += 1
            else:
                inventory -= tr.qty
                cash += tr.price * tr.qty
                n_fills_ask += 1
            n_fills += 1
            queue_ahead_fills.append(tr.maker_queue_ahead_at_submit)
            fill_waits.append(tr.t - tr.maker_t_submit)
            if tr.maker_order_id == bid_oid:
                bid_oid = None
            elif tr.maker_order_id == ask_oid:
                ask_oid = None
            max_abs_inv = max(max_abs_inv, abs(inventory))

    _requote()
    _record_path()
    next_decision = di
    next_sample = si
    while sim.t < h:
        sim.step()
        _drain_trades()
        if sim.t >= next_decision:
            _requote()
            _record_path()
            while next_decision <= sim.t:
                next_decision += di
        if sim.t >= next_sample:
            samples.append(sim.sample())
            while next_sample <= sim.t:
                next_sample += si

    final_mid = sim.mid if sim.mid is not None else last_mid
    final_mtm = cash + inventory * final_mid if final_mid is not None else float("nan")
    flow_diag: dict[str, Any] | None = None
    if len(signs) >= 52:
        flow_diag = regime_flow_diagnostics(signs)
    flow_report: dict[str, Any] | None = None
    if isinstance(flow, HawkesFlow):
        flow_report = {
            "kind": "hawkes_flow",
            "n_mo": flow.n_mo,
            "intensity_mult_final": float(flow.current().intensity_mult),
            "intensity_mult_bound": flow.intensity_mult_bound,
        }
    return {
        "label": "SYNTHETIC",
        "data_source": ZI_LOB_REVISION,
        "agent_revision": HAWKES_MM_REVISION,
        "research_only": True,
        "live_pnl_claim": False,
        "claim": "simulator_internal_diagnostic_only",
        "seed": config.seed,
        "horizon": h,
        "decision_interval": di,
        "inventory_cap": cap,
        "n_events": sim.n_events,
        "n_decisions": n_decisions,
        "n_fills": n_fills,
        "n_fills_bid": n_fills_bid,
        "n_fills_ask": n_fills_ask,
        "n_mm_cancels": n_cancels,
        "n_quote_clips": n_clips,
        "n_skipped_decisions": n_skipped,
        "inventory_final": inventory,
        "max_abs_inventory": max_abs_inv,
        "mean_abs_inventory": float(np.mean(np.abs(np.asarray(inv_path, dtype=np.float64))))
        if inv_path
        else float("nan"),
        "inventory_path": inv_path,
        "inventory_path_times": inv_times,
        "lam_plus_path": lam_p_path,
        "lam_minus_path": lam_m_path,
        # Simulator-internal mark-to-market accounting. Diagnostic only:
        # never a headline metric, never market evidence, no live-trading claim.
        "sim_internal_mtm_pnl_path": mtm_path,
        "sim_internal_mtm_pnl_final": float(final_mtm),
        "mean_queue_ahead_at_fill": float(np.mean(queue_ahead_fills))
        if queue_ahead_fills
        else float("nan"),
        "mean_fill_wait_seconds": float(np.mean(fill_waits)) if fill_waits else float("nan"),
        "phase_metrics": book_phase_metrics(samples),
        "flow_diagnostics": flow_diag,
        "flow_report": flow_report,
        "n_signs": len(signs),
        "event_counts": sim.event_counts(),
    }


# ---------------------------------------------------------------------------
# Sealed bench receipt (hawkes_mm.v1)
# ---------------------------------------------------------------------------


def _half_life_numeric(
    params: BivariateHawkesParams, *, dt: float = 1e-3, window: float | None = None
) -> dict[str, float]:
    """Measure the buy-impulse half-life on a propagated intensity path.

    One buy MO at t=0: excess intensity is ``alpha_pp * beta e^{-beta t}``
    whose theoretical half-life is ``ln 2 / beta``. The numeric value comes
    from :func:`hawkes_intensity_path` (the discrete convolution), so the
    residual also covers the binning error.
    """
    if window is None:
        window = 8.0 / params.beta
    times = np.arange(0.0, window, dt)
    path = hawkes_intensity_path(times, np.array([0.0]), np.array([]), params)
    excess = path["lam_plus"] - params.mu_plus
    peak = float(excess[0])
    target = 0.5 * peak
    idx = int(np.searchsorted(-excess, -target))
    if idx >= excess.size:
        idx = excess.size - 1
    hl_num = float(times[idx])
    hl_theory = math.log(2.0) / params.beta
    return {
        "impulse_half_life": hl_num,
        "impulse_half_life_theory": hl_theory,
        "impulse_half_life_residual": hl_num - hl_theory,
        "impulse_peak_excess": peak,
    }


def hawkes_mm_bench(
    *,
    seed: int = 0,
    quick: bool = True,
) -> dict[str, Any]:
    """Sealed ``hawkes_mm.v1`` bench receipt (SYNTHETIC, proper scores only).

    Blocks:

    - ``impulse_response``: buy-MO impulse half-life (numeric vs ``ln2/beta``).
    - ``poisson_limit``: alpha = 0 solve vs the Avellaneda-Stoikov closed
      form (``models.market_making.as_optimal_quotes``) — max absolute
      offset residual and relative half-spread error across inventory.
    - ``excitation``: spread-vs-excitation response at flat inventory —
      the log-log slope of the total quoted spread in ``lambda+`` and the
      ask-offset response to a buy-side excitation jump.
    - ``session``: one HawkesFlow-driven :func:`run_impulse_mm_session`
      bundle (bounded inventory; honesty keys).

    Sealed with :func:`quant_fund.utils.receipt.seal_receipt` (hash of the
    canonical payload) and stamped with ``git_revision``. Deterministic.
    """
    _ = _nonneg_finite(float(seed), "seed")
    # Lazy: models.market_making owns the AS closed form — the deferred
    # import is the sanctioned layer-order cycle-breaker (see
    # zi_lob_simulator.avellaneda_stoikov_quotes).
    from quant_fund.models.market_making import as_optimal_quotes

    params = BivariateHawkesParams(
        mu_plus=0.10,
        mu_minus=0.10,
        alpha_pp=0.45,
        alpha_pm=0.10,
        alpha_mp=0.10,
        alpha_mm=0.45,
        beta=0.8,
    )
    impulse = _half_life_numeric(params)

    gamma, sigma, kappa, a_fill = 0.05, 0.05, 500.0, 0.8
    horizon = 40.0 if quick else 120.0
    n_t, n_lam, lam_max = (120, 15, 1.6) if quick else (400, 25, 1.6)
    grid: dict[str, float | int] = {"n_t": n_t, "n_lam": n_lam, "lam_max": lam_max}

    # --- Poisson limit: alpha -> 0 recovers AS offsets -------------------
    poisson_cfg = ImpulseMMConfig(
        gamma=gamma,
        sigma=sigma,
        kappa=kappa,
        a_fill=a_fill,
        q_max=4,
        hawkes=BivariateHawkesParams(
            mu_plus=params.mu_plus,
            mu_minus=params.mu_minus,
            alpha_pp=0.0,
            alpha_pm=0.0,
            alpha_mp=0.0,
            alpha_mm=0.0,
            beta=params.beta,
        ),
        horizon=horizon,
        n_t=n_t,
        n_lam=n_lam,
        lam_max=lam_max,
    )
    sol0 = solve_impulse_mm(poisson_cfg)
    mid = 100.0
    tau_eval = 0.5 * horizon
    qs = (-2, -1, 0, 1, 2)
    resid: list[float] = []
    spread_err: list[float] = []
    for q in qs:
        as_q = as_optimal_quotes(mid, float(q), gamma, sigma, tau_eval, kappa)
        db, da = sol0.offsets(tau_eval, q, poisson_cfg.hawkes.mu_plus, poisson_cfg.hawkes.mu_minus)
        got_bid, got_ask = mid - db, mid + da
        resid.append(abs(got_bid - float(as_q["bid"])))
        resid.append(abs(got_ask - float(as_q["ask"])))
        as_spread = float(as_q["ask"]) - float(as_q["bid"])
        spread_err.append(abs((got_ask - got_bid) - as_spread) / as_spread)
    poisson = {
        "max_abs_offset_residual": float(max(resid)),
        "max_rel_spread_error": float(max(spread_err)),
        "n_q_points": len(qs),
        "tau_eval": tau_eval,
    }

    # --- Excitation response ---------------------------------------------
    sol_h = solve_impulse_mm(
        ImpulseMMConfig(
            gamma=gamma,
            sigma=sigma,
            kappa=kappa,
            a_fill=a_fill,
            q_max=4,
            hawkes=params,
            horizon=horizon,
            n_t=n_t,
            n_lam=n_lam,
            lam_max=lam_max,
        )
    )
    lams = np.linspace(params.mu_plus, params.stationary_mean[0] + 2.0 * params.jump_buy[0], 6)
    spreads = np.array(
        [
            sum(sol_h.offsets(tau_eval, 0, float(lp), params.mu_minus))
            if all(math.isfinite(v) for v in sol_h.offsets(tau_eval, 0, float(lp), params.mu_minus))
            else float("nan")
            for lp in lams
        ],
        dtype=np.float64,
    )
    finite = np.isfinite(spreads) & (spreads > 0.0)
    if int(finite.sum()) >= 3:
        slope = float(np.polyfit(np.log(lams[finite]), np.log(spreads[finite]), 1)[0])
    else:
        slope = float("nan")
    excitation = {
        "lam_plus_levels": [float(x) for x in lams],
        "spread_levels": [float(x) for x in spreads],
        "spread_excitation_loglog_slope": slope,
        "jump_buy": list(params.jump_buy),
        "jump_sell": list(params.jump_sell),
        "stationary_mean": list(params.stationary_mean),
    }

    # --- Session under HawkesFlow -----------------------------------------
    flow = HawkesFlow(excitation=0.35, decay_rho=0.90, p_buy=0.5)
    session = run_impulse_mm_session(
        config=ZILobConfig(
            s0=100.0,
            tick=0.01,
            lam=0.06,
            mu=0.10,
            theta_cxl=0.02,
            p_buy=0.5,
            band=5,
            init_levels=3,
            init_depth=5,
            seed=int(seed),
        ),
        solution=sol_h,
        horizon=600.0,
        decision_interval=1.0,
        inventory_cap=4,
        sample_interval=25.0,
        flow=flow,
        estimator=BivariateHawkes(params),
    )

    receipt = {
        "kind": "hawkes_mm.v1",
        "label": "SYNTHETIC",
        "data_source": ZI_LOB_REVISION,
        "agent_revision": HAWKES_MM_REVISION,
        "research_only": True,
        "live_pnl_claim": False,
        "claim": "simulator_internal_diagnostic_only",
        "git_revision": git_revision(),
        "seed": int(seed),
        "hawkes_params": {
            "mu_plus": params.mu_plus,
            "mu_minus": params.mu_minus,
            "alpha_pp": params.alpha_pp,
            "alpha_pm": params.alpha_pm,
            "alpha_mp": params.alpha_mp,
            "alpha_mm": params.alpha_mm,
            "beta": params.beta,
            "branching_ratio": params.branching_ratio,
        },
        "mm_params": {
            "gamma": gamma,
            "sigma": sigma,
            "kappa": kappa,
            "a_fill": a_fill,
            "horizon": horizon,
            "grid": dict(grid),
            "quick": bool(quick),
        },
        "impulse_response": impulse,
        "poisson_limit": poisson,
        "excitation": excitation,
        "session": session,
    }
    return seal_receipt(receipt)
