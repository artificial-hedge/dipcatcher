"""Liquidity tail risk in a sequential limit order book. No Sharpe.

**Labeled SYNTHETIC** research infrastructure (wave 20): the sequential-LOB
equilibrium model of Çetin, Lin & Livieri (2026), "When large trades are not
(automatically) news: liquidity tail risk and price discovery",
arXiv:2607.01198 [q-fin.TR] — a discrete-time limit order book with
asymmetric information in which liquidity suppliers observe *aggregate*
order flow ``Y_t = X*_t + Z_t``, never its decomposition into informed
demand ``X*_t`` and uninformed liquidity shocks ``Z_t``.

Model (paper Secs. 2-3): competitive liquidity suppliers post a
non-decreasing marginal price schedule ``h(y, t)`` indexed by cumulative
depth, so a market order of size ``x`` costs ``int_0^x h(u) du`` (eq. 2.2).
``N_t > 1`` risk-neutral myopic insiders know the fundamental ``V`` (support
``[m, M]``) and trade after noise traders; their symmetric first-order
condition (eq. 3.1) defines the marginal-cost schedule

    F(x, t) = E[ h(x + Z_t)/N_t + (N_t-1)/(N_t x) int_0^x h(u + Z_t) du ],  (3.2)

strictly increasing, with equilibrium informed demand ``X*_t = F^{-1}(V)``.
Zero-profit limit prices are Glosten (1994) tail expectations,
``h(y) = E[V | X* + Z >= y]`` for ``y > 0`` (eq. 3.4). Combining yields the
non-linear fixed point ``F = T_{t,nu} F`` (eqs. 3.8-3.12): with
``phi_g^pm(x)`` the convolution ratios of the posterior tail functionals
``Phi^pm``/``Pi^pm`` (eq. 3.5) against the noise kernel ``q_nu``,

    (T g)(x) = int [ q_nu(x-z)/N + (N-1)/N * q_bar_nu(x,z) ] phi_g(z) dz,

    q_bar_nu(x,z) = (1/x) int_0^x q_nu(u-z) du  (x != 0),  q_bar(0,z)=q(z).

Belief update (eq. 3.3): the posterior ``p_{t,V}(v)`` is proportional to the
prior times, for each past period ``s``, a Student-t likelihood factor
``{1 + (1/nu)[(Y_s - F_s^{-1}(v))/sigma]^2}^{-(nu+1)/2}`` — suppliers learn
``V`` only through the endogenous, dependent aggregate flow.

Paper results implemented as diagnostics (SYNTHETIC only):

- fixed-point existence inside the tail-controlled class ``K_{t,nu}``
  (Def. 3.2 / Thm. 3.1) — our iteration enforces the ``[m, M]`` bound and
  reports convergence residuals;
- posterior consistency (Sec. 4): posterior mean/standard deviation
  trajectories over repeated trading rounds;
- tail asymptotics (Thm. 5.1, Cor. 5.1): the recursion
  ``rho_t^+ = (psi'_t - 1)/(1 - psi'_t/N_t)`` with
  ``psi'_t = alpha_t/(alpha_t+1)``,
  ``alpha_t = kappa - (nu+1) * sum_{s<t} 1/rho_s``, plus an empirical
  log-log tail fit of ``M - F``;
- eventual informed-demand dominance (Prop. 5.1): the trade size where the
  informed share of marginal exceedance flow exceeds the noise share
  (crossover depth) — heavier ``nu`` tails push it farther out;
- eventual book monotonicity in the far tails (Prop. 5.2 / Cor. 5.3);
- persistent bid-ask spreads after large heavy-tailed trades: the spread
  ``h(0+) - h(0-) = phi^+(0) - phi^-(0) > 0`` (Lemma 3.1(iii)) is tracked
  across periods in the learning run.

Numerics follow the paper's Section 6 setup verbatim where stated: odd
x-grid centered at 0, trapezoidal quadrature, closed-form
``q_bar(x,z) = [F_nu(x-z) - F_nu(-z)]/x``, posterior weights on a v-grid
with linearly interpolated tail functionals, fixed-point iteration
terminated on a sup-norm criterion. One deviation, required by finite
grids: ``F`` is extended to the convolution ``z``-grid by clamping to its
endpoint values (F in ``[m,M]`` is bounded, so the extension stays inside
the class), and ``F^{-1}(v)`` beyond the grid is extrapolated by the fitted
regular-variation tail ``M - F ~ c(1+x)^rho`` — the same power law the
theory selects (Thm. 5.1), never an ad-hoc linear extension.

Honesty: every output is a SYNTHETIC correctness diagnostic, never market
evidence. Spreads, impact slopes, crossover depths, tail exponents and
posterior-consistency errors are equilibrium *diagnostics* of a simulated
model; no profit-and-loss quantity exists anywhere in this module, there is
no broker connectivity and no live-trading claim. All randomness flows
through seeded ``numpy.random.Generator`` streams; repeated calls are
bit-identical.

Composition notes (nothing reimplemented): ``microstructure.
zi_lob_simulator`` is a zero-intelligence Poisson-flow matching engine — a
different model class with no equilibrium asymmetric information and no
marginal-cost fixed point, so it shares no code path here.
``features/liquidity.glosten_harris`` is the Glosten-Harris (1988)
trade-indicator spread decomposition on transaction data, a different
object from this module's Glosten (1994) tail-expectation book.
``metrics/scoring.crps_student_t`` is a forecast scoring rule, unrelated to
the Student-t *noise kernel* used inside the equilibrium map. The
Student-t/Gaussian noise laws here are scipy location-scale primitives.

References:
- Çetin, Lin & Livieri (2026). When large trades are not (automatically)
  news: liquidity tail risk and price discovery. arXiv:2607.01198
  [q-fin.TR] — model, fixed point (3.10)-(3.12), posterior update (3.3),
  tail asymptotics Thm. 5.1 / Cor. 5.1, dominance Prop. 5.1, tail
  monotonicity Prop. 5.2 / Cor. 5.3, spread Lemma 3.1(iii), numerics Sec. 6.
- Çetin & Waelbroeck (2023). Power laws in market microstructure.
  *Frontiers of Mathematical Finance* 2(1):56-98 — static one-period
  Gaussian antecedent (paper ref. [6]).
- Glosten & Milgrom (1985). Bid, ask and transaction prices in a
  specialist market with heterogeneously informed traders. *JFE* 14(1):
  71-100 — sequential-trade Bayesian pricing logic (paper ref. [14]).
- Glosten (1994). Is the electronic open limit order book inevitable?
  *J. Finance* 49(4):1127-1161 — tail-expectation limit prices (ref. [15]).
- Kyle (1985). Continuous auctions and insider trading. *Econometrica*
  53(6):1315-1335 — camouflage intuition (ref. [21]).
- Ozsoylev & Takayama (2010). Price, trade size, and information
  revelation in multi-period securities markets. *J. Financial Markets*
  13(1):49-76 — asymptotic learning benchmark (ref. [26]).
- Shalizi (2009). Dynamics of Bayesian updating with dependent data and
  misspecified models. *Electronic J. Statistics* 3:1039-1074 —
  posterior-consistency machinery (ref. [32]).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm as _norm
from scipy.stats import t as _st_t

Array = NDArray[np.float64]

NoiseKind = Literal["student_t", "gaussian"]
Side = Literal["ask", "bid"]

LTL_REVISION = "SYNTHETIC_LIQUIDITY_TAIL_LOB_v1"


class FixedPointError(RuntimeError):
    """Raised by strict accessors when the marginal-cost fixed point did not converge."""


# ---------------------------------------------------------------------------
# Fail-closed validation helpers (house style: mirroring zi_lob_simulator)
# ---------------------------------------------------------------------------


def _pos_finite(x: float, name: str) -> float:
    v = float(x)
    if not math.isfinite(v) or v <= 0.0:
        raise ValueError(f"{name} must be positive and finite, got {x!r}")
    return v


def _odd_int(x: int, name: str, lo: int) -> int:
    if isinstance(x, bool) or int(x) != x or int(x) < lo or int(x) % 2 == 0:
        raise ValueError(f"{name} must be an odd integer >= {lo}, got {x!r}")
    return int(x)


def _grid_1d(values: object, name: str, min_size: int = 2) -> Array:
    v = np.asarray(values, dtype=float).reshape(-1)
    if v.size < min_size:
        raise ValueError(f"{name} must have at least {min_size} points, got {v.size}")
    if not bool(np.all(np.isfinite(v))):
        raise ValueError(f"{name} must be finite")
    if not bool(np.all(np.diff(v) > 0.0)):
        raise ValueError(f"{name} must be strictly increasing")
    return v


def _resolve_rng(rng: np.random.Generator | int) -> np.random.Generator:
    if isinstance(rng, np.random.Generator):
        return rng
    if isinstance(rng, bool):
        raise ValueError("rng seed must be an int or np.random.Generator")
    return np.random.default_rng(int(rng))


# ---------------------------------------------------------------------------
# Noise specification: Student-t (liquidity tail risk) or Gaussian benchmark
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NoiseSpec:
    """Uninformed aggregate order-flow law ``Z_t`` (paper Remark 2.1).

    ``kind="student_t"`` is the paper's heavy-tailed specification
    ``Z ~ T_nu(0, sigma)`` with tail index ``nu`` — the *liquidity tail risk*
    state variable; smaller ``nu`` means heavier tails. ``kind="gaussian"``
    is the thin-tailed benchmark ``Z ~ N(0, sigma^2)`` of Çetin & Waelbroeck
    (2023), stored as ``nu = +inf``. Fail-closed on nonpositive tail index,
    nonpositive scale, or an inconsistent kind/nu combination.
    """

    kind: NoiseKind
    nu: float
    sigma: float

    def __post_init__(self) -> None:
        if self.kind not in ("student_t", "gaussian"):
            raise ValueError(f"kind must be 'student_t' or 'gaussian', got {self.kind!r}")
        object.__setattr__(self, "sigma", _pos_finite(self.sigma, "sigma"))
        nu = float(self.nu)
        if self.kind == "student_t":
            if not math.isfinite(nu) or nu <= 0.0:
                raise ValueError(f"tail index nu must be positive and finite, got {self.nu!r}")
        elif nu != math.inf:
            raise ValueError("gaussian noise must carry nu = math.inf")
        object.__setattr__(self, "nu", nu)

    def pdf(self, z: Array) -> Array:
        """Density ``q_nu(z; 0, sigma)`` (paper eq. 2.3)."""
        zz = np.asarray(z, dtype=float)
        if self.kind == "student_t":
            return np.asarray(_st_t.pdf(zz / self.sigma, self.nu) / self.sigma, dtype=float)
        return np.asarray(_norm.pdf(zz / self.sigma) / self.sigma, dtype=float)

    def logpdf(self, z: Array) -> Array:
        """Log density — used for the log-space belief update (eq. 3.3)."""
        zz = np.asarray(z, dtype=float)
        if self.kind == "student_t":
            return np.asarray(
                _st_t.logpdf(zz / self.sigma, self.nu) - math.log(self.sigma), dtype=float
            )
        return np.asarray(_norm.logpdf(zz / self.sigma) - math.log(self.sigma), dtype=float)

    def cdf(self, z: Array) -> Array:
        zz = np.asarray(z, dtype=float)
        if self.kind == "student_t":
            return np.asarray(_st_t.cdf(zz / self.sigma, self.nu), dtype=float)
        return np.asarray(_norm.cdf(zz / self.sigma), dtype=float)

    def sf(self, z: Array) -> Array:
        """Survival ``P(Z >= z)`` — the noise-side exceedance of the crossover diagnostic."""
        zz = np.asarray(z, dtype=float)
        if self.kind == "student_t":
            return np.asarray(_st_t.sf(zz / self.sigma, self.nu), dtype=float)
        return np.asarray(_norm.sf(zz / self.sigma), dtype=float)

    def sample(self, rng: np.random.Generator, size: int) -> Array:
        """Seeded draws ``Z_t``; ``standard_t(nu)`` matches ``T_nu(0,1)`` exactly."""
        if self.kind == "student_t":
            return np.asarray(self.sigma * rng.standard_t(self.nu, size=size), dtype=float)
        return np.asarray(self.sigma * rng.standard_normal(size), dtype=float)


def student_t_noise(nu: float, sigma: float) -> NoiseSpec:
    """Validated constructor for the heavy-tailed noise law ``T_nu(0, sigma)``."""
    return NoiseSpec(kind="student_t", nu=float(nu), sigma=float(sigma))


def gaussian_noise(sigma: float) -> NoiseSpec:
    """Validated constructor for the thin-tailed benchmark ``N(0, sigma^2)``."""
    return NoiseSpec(kind="gaussian", nu=math.inf, sigma=float(sigma))


# ---------------------------------------------------------------------------
# Beliefs: prior/posterior over V on a finite grid + tail functionals (3.5)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Belief:
    """Atomless-on-a-grid belief over the fundamental ``V`` (Assumption 3.2).

    ``v`` is a strictly increasing grid covering the support ``[m, M]``;
    ``w`` are normalized masses (``w_j ~ p_V(v_j) dv``). Precomputed tail
    functionals of eq. (3.5): ``Pi+`` (P[V >= y]), ``Phi+`` (E[V 1{V>=y}]),
    and the bid-side ``Pi-``/``Phi-``, each represented on the ``v``-grid and
    linearly interpolated for off-grid arguments — exactly the paper's
    Section 6 discretization. Endpoint conventions of eq. (3.6):
    ``psi+(M) = M`` and ``psi-(m) = m`` where the tail mass vanishes.
    """

    v: Array
    w: Array
    _pi_plus: Array = field(init=False, repr=False)
    _phi_plus: Array = field(init=False, repr=False)
    _pi_minus: Array = field(init=False, repr=False)
    _phi_minus: Array = field(init=False, repr=False)

    def __post_init__(self) -> None:
        v = _grid_1d(self.v, "v")
        w = np.asarray(self.w, dtype=float).reshape(-1)
        if w.shape != v.shape:
            raise ValueError("w must have the same length as v")
        if not bool(np.all(np.isfinite(w))):
            raise ValueError("w must be finite")
        if bool(np.any(w < 0.0)):
            raise ValueError("w must be non-negative")
        total = float(w.sum())
        if not math.isfinite(total) or total <= 0.0:
            raise ValueError("w must carry positive total mass")
        w = w / total
        object.__setattr__(self, "v", v)
        object.__setattr__(self, "w", w)
        # Tail functionals on the v-grid (eq. 3.5), cumulative from the top.
        rev_w = w[::-1]
        rev_vw = (v * w)[::-1]
        pi_plus = np.cumsum(rev_w)[::-1]
        phi_plus = np.cumsum(rev_vw)[::-1]
        pi_minus = np.cumsum(w)
        phi_minus = np.cumsum(v * w)
        object.__setattr__(self, "_pi_plus", pi_plus)
        object.__setattr__(self, "_phi_plus", phi_plus)
        object.__setattr__(self, "_pi_minus", pi_minus)
        object.__setattr__(self, "_phi_minus", phi_minus)

    @property
    def m(self) -> float:
        return float(self.v[0])

    @property
    def M(self) -> float:
        return float(self.v[-1])

    @property
    def mean(self) -> float:
        return float(np.sum(self.v * self.w))

    @property
    def var(self) -> float:
        mu = self.mean
        return float(np.sum((self.v - mu) ** 2 * self.w))

    @property
    def std(self) -> float:
        return float(math.sqrt(max(self.var, 0.0)))

    def mass_within(self, v0: float, eps: float) -> float:
        """P(|V - v0| <= eps) — the posterior concentration diagnostic of Sec. 4."""
        e = _pos_finite(eps, "eps")
        return float(np.sum(self.w[np.abs(self.v - float(v0)) <= e]))

    def tail_plus(self, y: Array) -> tuple[Array, Array]:
        """(Phi+(y), Pi+(y)) interpolated; ``y < m`` -> (E[V], 1), ``y >= M`` -> (0, 0)."""
        yy = np.asarray(y, dtype=float)
        pi = np.interp(yy, self.v, self._pi_plus, left=1.0, right=0.0)
        phi = np.interp(yy, self.v, self._phi_plus, left=self.mean, right=0.0)
        return np.asarray(phi, dtype=float), np.asarray(pi, dtype=float)

    def tail_minus(self, y: Array) -> tuple[Array, Array]:
        """(Phi-(y), Pi-(y)) interpolated; ``y <= m`` -> (0, 0), ``y > M`` -> (E[V], 1)."""
        yy = np.asarray(y, dtype=float)
        pi = np.interp(yy, self.v, self._pi_minus, left=0.0, right=1.0)
        phi = np.interp(yy, self.v, self._phi_minus, left=0.0, right=self.mean)
        return np.asarray(phi, dtype=float), np.asarray(pi, dtype=float)

    def psi_plus(self, y: Array) -> Array:
        """Tail expectation E[V | V >= y] (eq. 3.6); ``M`` where Pi+ vanishes."""
        phi, pi = self.tail_plus(np.asarray(y, dtype=float))
        out = np.where(pi > 0.0, phi / np.where(pi > 0.0, pi, 1.0), self.M)
        return np.asarray(out, dtype=float)

    def psi_minus(self, y: Array) -> Array:
        """Tail expectation E[V | V <= y] (eq. 3.6); ``m`` where Pi- vanishes."""
        phi, pi = self.tail_minus(np.asarray(y, dtype=float))
        out = np.where(pi > 0.0, phi / np.where(pi > 0.0, pi, 1.0), self.m)
        return np.asarray(out, dtype=float)

    def posterior(self, log_lik_increment: Array) -> Belief:
        """Bayesian update in log space (eq. 3.3): ``w ∝ w * exp(log lik)``.

        Grid points with zero prior mass stay zero (prior support is
        preserved); -inf log-likelihoods kill grid points. Fail-closed when
        every retained point underflows — the posterior is undefined there.
        """
        ll = np.asarray(log_lik_increment, dtype=float).reshape(-1)
        if ll.shape != self.v.shape:
            raise ValueError("log_lik_increment must have one entry per v-grid point")
        with np.errstate(divide="ignore"):
            logw = np.log(self.w) + ll
        if not bool(np.any(np.isfinite(logw))):
            raise ValueError(
                "posterior collapsed: every grid point has -inf log weight "
                "(likelihood underflow — check the noise scale vs support width)"
            )
        logw = logw - float(np.max(logw[np.isfinite(logw)]))
        w = np.where(np.isfinite(logw), np.exp(logw), 0.0)
        return Belief(v=self.v, w=np.asarray(w, dtype=float))


def grid_belief(v: object, density: object) -> Belief:
    """Belief from density values on a grid (trapezoid-normalized).

    Fail-closed: non-finite or negative densities, or a vanishing total
    mass — a degenerate prior defines no beliefs.
    """
    vv = _grid_1d(v, "v")
    d = np.asarray(density, dtype=float).reshape(-1)
    if d.shape != vv.shape:
        raise ValueError("density must have one value per v-grid point")
    if not bool(np.all(np.isfinite(d))) or bool(np.any(d < 0.0)):
        raise ValueError("density must be finite and non-negative")
    dv = np.diff(vv)
    w = np.empty(vv.size, dtype=float)
    w[0] = 0.5 * d[0] * dv[0]
    w[-1] = 0.5 * d[-1] * dv[-1]
    w[1:-1] = 0.5 * d[1:-1] * (dv[1:] + dv[:-1])
    if float(w.sum()) <= 0.0:
        raise ValueError("density integrates to zero on the grid — degenerate prior")
    return Belief(v=vv, w=w)


def uniform_belief(m: float, M: float, n_v: int = 501) -> Belief:
    """Uniform prior on ``[m, M]`` (paper Sec. 6 bounded-support spec (i))."""
    mm, MM = float(m), float(M)
    if not (math.isfinite(mm) and math.isfinite(MM)) or not mm < MM:
        raise ValueError(f"uniform prior needs m < M finite, got [{m!r}, {M!r}]")
    n = int(n_v)
    if n < 21:
        raise ValueError("n_v must be >= 21")
    v = np.linspace(mm, MM, n)
    dv = (MM - mm) / (n - 1)
    w = np.full(n, dv)
    w[0] = w[-1] = 0.5 * dv
    return Belief(v=v, w=w)


def beta_belief(a: float, b: float, m: float, M: float, n_v: int = 501) -> Belief:
    """Beta(a,b) prior scaled to ``[m, M]`` (Sec. 6 spec (i); endpoint exponent kappa = b)."""
    aa, bb = _pos_finite(a, "a"), _pos_finite(b, "b")
    mm, MM = float(m), float(M)
    if not (math.isfinite(mm) and math.isfinite(MM)) or not mm < MM:
        raise ValueError(f"beta prior needs m < M finite, got [{m!r}, {M!r}]")
    n = int(n_v)
    if n < 21:
        raise ValueError("n_v must be >= 21")
    from scipy.stats import beta as _beta

    v = np.linspace(mm, MM, n)
    u = (v - mm) / (MM - mm)
    with np.errstate(divide="ignore"):
        d = _beta.pdf(u, aa, bb) / (MM - mm)
    d = np.where(np.isfinite(d), d, 0.0)
    return grid_belief(v, np.asarray(d, dtype=float))


def truncnorm_belief(mu: float, sd: float, m: float, M: float, n_v: int = 501) -> Belief:
    """Truncated-normal prior on ``[m, M]`` (Sec. 6 spec (ii))."""
    sd = _pos_finite(sd, "sd")
    mm, MM = float(m), float(M)
    if not (math.isfinite(mm) and math.isfinite(MM)) or not mm < MM:
        raise ValueError(f"truncnorm prior needs m < M finite, got [{m!r}, {M!r}]")
    n = int(n_v)
    if n < 21:
        raise ValueError("n_v must be >= 21")
    from scipy.stats import truncnorm as _tn

    v = np.linspace(mm, MM, n)
    d = _tn.pdf(v, (mm - mu) / sd, (MM - mu) / sd, loc=mu, scale=sd)
    return grid_belief(v, np.asarray(d, dtype=float))


# ---------------------------------------------------------------------------
# Solver configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SolverConfig:
    """Discretization + iteration controls (paper Sec. 6 numerical setup).

    ``x``-grid: ``n_x`` odd points on ``[-x_max, x_max]`` carrying the
    marginal-cost schedule ``F`` and the price schedule ``h``. ``z``-grid:
    ``n_z`` odd points on ``[-z_max, z_max]`` for the convolutions (must
    strictly contain the x-grid so the kernel mass is interior).
    ``n_informed = N_t > 1`` (Assumption: deterministic, common knowledge —
    Remark 2.2). Iteration ``F <- (1-damping) F + damping * T F`` stops at
    sup-norm ``tol`` or ``max_iter``.
    """

    n_informed: int = 2
    x_max: float = 10.0
    n_x: int = 101
    z_max: float = 30.0
    n_z: int = 301
    tol: float = 1e-9
    max_iter: int = 500
    damping: float = 1.0

    def __post_init__(self) -> None:
        n = int(self.n_informed)
        if isinstance(self.n_informed, bool) or n < 2:
            raise ValueError(f"n_informed must be an integer > 1, got {self.n_informed!r}")
        object.__setattr__(self, "n_informed", n)
        object.__setattr__(self, "x_max", _pos_finite(self.x_max, "x_max"))
        object.__setattr__(self, "z_max", _pos_finite(self.z_max, "z_max"))
        object.__setattr__(self, "n_x", _odd_int(self.n_x, "n_x", 21))
        object.__setattr__(self, "n_z", _odd_int(self.n_z, "n_z", 21))
        if self.z_max <= self.x_max:
            raise ValueError(
                f"z_max ({self.z_max}) must exceed x_max ({self.x_max}) so the "
                "convolution kernel's mass is interior to the z-grid"
            )
        object.__setattr__(self, "tol", _pos_finite(self.tol, "tol"))
        mi = int(self.max_iter)
        if isinstance(self.max_iter, bool) or mi < 1:
            raise ValueError(f"max_iter must be an integer >= 1, got {self.max_iter!r}")
        object.__setattr__(self, "max_iter", mi)
        d = float(self.damping)
        if not math.isfinite(d) or not 0.0 < d <= 1.0:
            raise ValueError(f"damping must be in (0, 1], got {self.damping!r}")
        object.__setattr__(self, "damping", d)

    @property
    def x(self) -> Array:
        return np.linspace(-self.x_max, self.x_max, self.n_x)

    @property
    def z(self) -> Array:
        return np.linspace(-self.z_max, self.z_max, self.n_z)


# ---------------------------------------------------------------------------
# Empirical power tail (regular-variation diagnostic, Thm. 5.1 / Def. 3.2)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PowerTail:
    """Log-log fit ``M - F ~ c(1+x)^rho`` (ask) or ``F - m ~ c(1+|x|)^rho`` (bid).

    ``rho`` is the empirical regular-variation index (expected in (-1, 0)
    per Lemma 3.3 / Thm. 5.1); ``log_c`` the log coefficient. ``inverse``
    maps a fundamental value near the endpoint back to an order size —
    the tail-controlled extension of ``F^{-1}`` beyond the grid.
    """

    rho: float
    log_c: float
    rms_resid: float

    def inverse(self, gap: Array) -> Array:
        """|x| such that ``gap ~ c(1+|x|)^rho``; gap = M - v (ask) or v - m (bid).

        ``gap <= 0`` maps to ``+inf`` (demand unbounded at the endpoint —
        callers clip); the divide is silenced, not hidden.
        """
        g = np.asarray(gap, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            out = np.exp((np.log(g) - self.log_c) / self.rho) - 1.0
        return np.asarray(np.where(g > 0.0, out, np.inf), dtype=float)


def _fit_power_tail(x_abs: Array, gap: Array, frac_lo: float = 0.4) -> PowerTail | None:
    """Fit ``log gap ~ log c + rho log(1+|x|)`` on the outer window ``frac_lo * max``.

    Returns ``None`` when the window is degenerate (non-finite fit or a
    non-negative slope — no power-law decay to invert); callers then fall
    back to grid-edge clipping, which is documented in the module docstring.
    """
    if x_abs.size < 8:
        return None
    hi = float(np.max(x_abs))
    mask = (x_abs >= frac_lo * hi) & (x_abs <= 0.98 * hi) & (gap > 0.0)
    if int(np.count_nonzero(mask)) < 5:
        return None
    lx = np.log1p(x_abs[mask])
    lg = np.log(gap[mask])
    rho, log_c = np.polyfit(lx, lg, 1)
    if not (math.isfinite(rho) and math.isfinite(log_c)) or rho >= -1e-9:
        return None
    resid = lg - (log_c + rho * lx)
    rms = float(np.sqrt(np.mean(resid * resid)))
    return PowerTail(rho=float(rho), log_c=float(log_c), rms_resid=rms)


# ---------------------------------------------------------------------------
# Fixed-point solution object
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MarginalCostSolution:
    """Fixed-point output ``F*``, its price schedule ``h = phi_F`` (eq. 3.9).

    ``h_plus``/``h_minus`` keep the ask/bid branches separately so the spread
    ``h(0+) - h(0-)`` (Lemma 3.1(iii)) is available; ``h`` is the merged
    schedule of eq. (3.9) (``h_plus`` for ``x >= 0``, ``h_minus`` for
    ``x < 0``). ``converged`` records whether the sup-norm criterion was met;
    all downstream quantities go through :meth:`require_solution`, the strict
    accessor — an unconverged iterate is never exposed as an equilibrium.
    """

    x: Array
    F: Array
    h: Array
    h_plus: Array
    h_minus: Array
    converged: bool
    n_iter: int
    max_resid: float
    belief: Belief
    noise: NoiseSpec
    config: SolverConfig
    tail_plus_fit: PowerTail | None
    tail_minus_fit: PowerTail | None

    def require_solution(self) -> MarginalCostSolution:
        """Strict accessor: raise unless the iteration actually converged."""
        if not self.converged:
            raise FixedPointError(
                f"marginal-cost fixed point did not converge: "
                f"max_resid={self.max_resid:.3e} after {self.n_iter} iterations"
            )
        return self

    def _z_informed(self, v: Array) -> Array:
        """Core inversion ``x = F^{-1}(v)`` on the grid + power-tail extension.

        ``v`` inside ``(F(x_min), F(x_max))`` interpolates the monotone
        branch; beyond it, the fitted regular-variation tail inverts
        ``M - v ~ c(1+x)^rho`` (ask) / ``v - m ~ c(1+|x|)^rho`` (bid), the
        Thm. 5.1 tail shape — clipped at ``4 * x_max`` to keep the
        likelihood evaluation finite and honest. ``v >= M`` / ``v <= m``
        return ``+inf`` / ``-inf`` (demand unbounded at the endpoint).
        """
        self.require_solution()
        vv = np.asarray(v, dtype=float)
        out = np.interp(vv, self.F, self.x)
        m, M = self.belief.m, self.belief.M
        xmax = float(self.x[-1])
        above = vv > self.F[-1]
        below = vv < self.F[0]
        if bool(np.any(above)):
            gap = M - vv[above]
            if self.tail_plus_fit is not None:
                xa = self.tail_plus_fit.inverse(gap)
            else:
                xa = xmax * (1.0 + np.divide(self.F[-1] - vv[above], max(M - self.F[-1], 1e-12)))
            out[above] = np.clip(np.asarray(xa, dtype=float), xmax, 4.0 * xmax)
        if bool(np.any(below)):
            gap = vv[below] - m
            if self.tail_minus_fit is not None:
                xb = self.tail_minus_fit.inverse(gap)
            else:
                xb = xmax * (1.0 + np.divide(vv[below] - self.F[0], max(self.F[0] - m, 1e-12)))
            out[below] = -np.clip(np.asarray(xb, dtype=float), xmax, 4.0 * xmax)
        return np.asarray(out, dtype=float)

    def informed_demand(self, v: float | Array) -> Array:
        """Equilibrium informed demand ``X* = F^{-1}(v)`` (Def. 3.1(ii)).

        Scalars and arrays both return ``Array``; use ``float(...)`` at call
        sites needing a scalar. Fail-closed on ``v`` outside ``[m, M]``.
        """
        vv = np.asarray(v, dtype=float)
        if not bool(np.all(np.isfinite(vv))):
            raise ValueError("v must be finite")
        if bool(np.any(vv > self.belief.M)) or bool(np.any(vv < self.belief.m)):
            raise ValueError(f"v must lie in the support [{self.belief.m}, {self.belief.M}]")
        out = np.where(
            vv >= self.belief.M,
            np.inf,
            np.where(vv <= self.belief.m, -np.inf, self._z_informed(vv)),
        )
        return np.asarray(out, dtype=float)

    def price(self, y: float | Array) -> Array:
        """The posted LOB price ``h(y)`` (eq. 3.9) interpolated on the x-grid."""
        self.require_solution()
        return np.asarray(
            np.interp(np.asarray(y, dtype=float), self.x, self.h, left=self.h[0], right=self.h[-1]),
            dtype=float,
        )

    def spread(self) -> float:
        """Bid-ask spread ``h(0+) - h(0-) > 0`` (Lemma 3.1(iii))."""
        self.require_solution()
        i0 = self.config.n_x // 2
        return float(self.h_plus[i0] - self.h_minus[i0])

    # -- tail-risk diagnostics (Sec. 5) --------------------------------------

    def informed_exceedance(self, y: float | Array, side: Side = "ask") -> Array:
        """P(X* >= y) = Pi+(F(y)) (ask) / P(X* <= y) = Pi-(F(y)) (bid).

        Corollary 5.1: this is the informed side of the marginal exceedance
        flow at depth ``y``, regularly varying with the induced exponent.
        """
        self.require_solution()
        yy = np.asarray(y, dtype=float)
        fy = np.interp(yy, self.x, self.F, left=self.F[0], right=self.F[-1])
        if side == "ask":
            _, pi = self.belief.tail_plus(fy)
        elif side == "bid":
            _, pi = self.belief.tail_minus(fy)
        else:
            raise ValueError(f"side must be 'ask' or 'bid', got {side!r}")
        return np.asarray(pi, dtype=float)

    def informed_share(self, y: float | Array) -> Array:
        """Informed share of *marginal* order flow at depth ``y``:

            share(y) = f_{X*}(y) / (f_{X*}(y) + q(y)),

        where ``f_{X*}(y) = -d/dy P(X* >= y)`` on the ask side and
        ``+d/dy P(X* <= y)`` on the bid side — the density of informed
        demand at that depth — and ``q`` is the uninformed density. Because
        ``P(X* >= y) = Pi+(F(y))`` inherits F's regular variation
        (Cor. 5.1) while ``q`` decays at the noise tail rate, the share
        crosses 0.5 deeper under heavier uninformed tails — the paper's
        crossover diagnostic (Sec. 5).
        """
        self.require_solution()
        yy = np.asarray(y, dtype=float)
        x = self.x
        ask = x >= 0.0
        f_x = np.empty_like(x)
        pi_p = self.belief.tail_plus(self.F[ask])[1]
        f_x[ask] = -np.gradient(pi_p, x[ask])
        pi_m = self.belief.tail_minus(self.F[~ask])[1]
        f_x[~ask] = np.gradient(pi_m, x[~ask])
        np.clip(f_x, 0.0, None, out=f_x)
        fx = np.interp(yy, x, f_x)
        q = self.noise.pdf(yy)
        denom = fx + q
        return np.asarray(
            np.where(denom > 0.0, fx / np.where(denom > 0.0, denom, 1.0), 0.0),
            dtype=float,
        )

    def crossover_depth(self, threshold: float = 0.5) -> float:
        """Smallest ask depth where the informed share exceeds ``threshold``.

        The paper's "crossover-depth": the order size at which a trade
        becomes more likely information-driven than liquidity-driven.
        ``+inf`` when no grid depth crosses (honest non-detection — the
        ambiguity region extends beyond the grid, which heavy tails enlarge).
        """
        self.require_solution()
        t = float(threshold)
        if not math.isfinite(t) or not 0.0 < t < 1.0:
            raise ValueError("threshold must be in (0, 1)")
        y = self.x[self.x > 0.0]
        share = self.informed_share(y)
        hit = np.nonzero(share > t)[0]
        if hit.size == 0:
            return float("inf")
        return float(y[int(hit[0])])

    def conditional_informed_share(self, y: float) -> float:
        """Prop. 5.1's tilted posterior pi_y([y, inf)) = P(X* >= y | X* + Z >= y).

        Computed on the belief grid: numerator
        ``sum_j w_j 1{X*(v_j) >= y} sf(y - X*(v_j))`` over
        ``sum_j w_j sf(y - X*(v_j))``.
        """
        self.require_solution()
        y = float(y)
        if not math.isfinite(y) or y <= 0.0:
            raise ValueError("y must be positive and finite")
        xs = self._z_informed(self.belief.v)
        tail = self.noise.sf(y - xs)
        den = float(np.sum(self.belief.w * tail))
        num = float(np.sum(self.belief.w * tail * (xs >= y)))
        if den <= 0.0:
            raise ValueError("P(X* + Z >= y) is numerically zero at this depth")
        return num / den

    def empirical_tail_exponent(self, side: Side = "ask") -> float:
        """Empirical regular-variation index rho_hat of the fixed-point branch.

        Log-log slope of ``M - F`` (ask) or ``F - m`` (bid) over the outer
        window — the numeric counterpart of Thm. 5.1's ``rho_t^pm``.
        """
        self.require_solution()
        fit = self.tail_plus_fit if side == "ask" else self.tail_minus_fit
        if fit is None:
            raise ValueError("no valid power-tail fit on this branch (degenerate tail)")
        return float(fit.rho)

    def far_tail_monotone(self, frac: float = 0.7) -> bool:
        """Cor. 5.3 check: h increasing throughout the outer ``frac`` of the grid.

        Finite-difference slopes of ``h`` on ``x >= frac * x_max`` and
        ``x <= -frac * x_max`` must all be strictly positive (with a small
        numerical tolerance); this is the paper's recovered-in-the-tails
        book monotonicity.
        """
        self.require_solution()
        f = float(frac)
        if not 0.0 < f < 1.0:
            raise ValueError("frac must be in (0, 1)")
        dh = np.diff(self.h)
        xc = 0.5 * (self.x[:-1] + self.x[1:])
        mask = np.abs(xc) >= f * self.config.x_max
        if int(np.count_nonzero(mask)) < 2:
            raise ValueError("outer window too small for a tail slope check")
        return bool(np.all(dh[mask] > -1e-12))


# ---------------------------------------------------------------------------
# The Student-t fixed-point operator (eqs. 3.8-3.12)
# ---------------------------------------------------------------------------


def solve_marginal_cost(
    belief: Belief,
    noise: NoiseSpec,
    config: SolverConfig | None = None,
    F0: Array | None = None,
) -> MarginalCostSolution:
    """Solve ``F = T_{t,nu} F`` (eq. 3.12) by damped fixed-point iteration.

    Initialization defaults to the paper's central monotone start,
    ``F^(0)(x) = m + (M-m)(1 + tanh(x/(2 sigma)))/2`` (Sec. 6). Iterates are
    clipped into ``[m, M]`` — the bounded class of Def. 3.2(i) — and the
    loop stops on the sup-norm criterion ``||F^{k+1} - F^k||_inf <= tol`` or
    exhausts ``max_iter`` (``converged=False``; strict accessors then raise).
    Deterministic: no randomness anywhere.
    """
    cfg = config if config is not None else SolverConfig()
    x = cfg.x
    z = cfg.z
    i0 = cfg.n_x // 2
    dx = float(x[1] - x[0])
    dz = float(z[1] - z[0])
    n = cfg.n_informed
    m, M = belief.m, belief.M

    # Kernel matrices (trapezoid weights folded into dz / dx constants), each
    # row renormalized to a proper probability weighting: plain rectangle-rule
    # sums over the sharply peaked kernels (sigma ~ 0.1 << dx) exceed 1 by a
    # few percent, which would let the convex-combination structure of the map
    # push iterates outside [m, M] and create spurious absorbing plateaus.
    # Row normalization keeps every expectation an exact average, matching the
    # continuous model where int q dz = 1.
    K_z = noise.pdf(x[:, None] - z[None, :]) * dz  # (n_x, n_z): int q(x-z) f(z) dz
    K_x = noise.pdf(x[:, None] - x[None, :]) * dx  # (n_x, n_x): G = q * h on x-grid
    K_z /= K_z.sum(axis=1, keepdims=True)
    K_x /= K_x.sum(axis=1, keepdims=True)

    def operator(Fx: Array) -> tuple[Array, Array, Array, Array]:
        # extend F to the z-grid by endpoint clamping (bounded class, Def. 3.2(i))
        F_z = np.interp(z, x, Fx, left=Fx[0], right=Fx[-1])
        phi_p, pi_p = belief.tail_plus(F_z)
        phi_m, pi_m = belief.tail_minus(F_z)
        # phi_F evaluated on the x-grid: convolution-ratio of eq. (3.8)
        num_px = K_z @ phi_p
        den_px = K_z @ pi_p
        num_mx = K_z @ phi_m
        den_mx = K_z @ pi_m
        phi_plus_x = np.where(den_px > 0.0, num_px / np.where(den_px > 0.0, den_px, 1.0), M)
        phi_minus_x = np.where(den_mx > 0.0, num_mx / np.where(den_mx > 0.0, den_mx, 1.0), m)
        h_x = np.where(x >= 0.0, phi_plus_x, phi_minus_x)  # eq. (3.9)
        h_x = np.clip(h_x, m, M)
        G = K_x @ h_x  # (q_nu * h)(x)
        # H(x) = (1/x) int_0^x G(u) du via cumulative trapezoid about index i0;
        # H(0) = G(0) is the continuous extension (eq. 3.11 at x = 0).
        seg_g = 0.5 * (G[:-1] + G[1:]) * dx
        cum_g = np.concatenate([[0.0], np.cumsum(seg_g)])
        integ = cum_g - cum_g[i0]
        H = np.where(x != 0.0, integ / np.where(x != 0.0, x, 1.0), 0.0)
        H[i0] = G[i0]
        # T F = G/N + (N-1)/N * H  (eq. 3.15), clipped into [m, M] (Def. 3.2(i)).
        F_new = np.clip((G + (n - 1.0) * H) / n, m, M)
        return np.asarray(F_new, dtype=float), phi_plus_x, phi_minus_x, h_x

    if F0 is None:
        sigma0 = noise.sigma
        F = m + (M - m) * 0.5 * (1.0 + np.tanh(x / (2.0 * sigma0)))
    else:
        F = np.asarray(F0, dtype=float).reshape(-1)
        if F.shape != x.shape:
            raise ValueError(f"F0 must have one value per x-grid point ({x.size})")
        if not bool(np.all(np.isfinite(F))):
            raise ValueError("F0 must be finite")
        F = np.clip(F, m, M)

    converged = False
    n_iter = 0
    resid = float("inf")
    h = np.full(cfg.n_x, 0.5 * (m + M))
    h_plus = h.copy()
    h_minus = h.copy()
    for _ in range(cfg.max_iter):
        n_iter += 1
        F_new, h_plus, h_minus, h = operator(F)
        F_next = cfg.damping * F_new + (1.0 - cfg.damping) * F
        resid = float(np.max(np.abs(F_next - F)))
        F = F_next
        if resid <= cfg.tol:
            converged = True
            break
    # Report the residual of the map itself, ||T F - F||_inf, on the final F.
    F_map, h_plus, h_minus, h = operator(F)
    max_resid = float(np.max(np.abs(F_map - F)))

    # Empirical power tails (outer window) for diagnostics + tail inversion.
    pos = x > 0.0
    tail_p = _fit_power_tail(x[pos], M - F[pos])
    tail_m = _fit_power_tail(-x[~pos], F[~pos] - m)
    return MarginalCostSolution(
        x=x,
        F=np.asarray(F, dtype=float),
        h=np.asarray(h, dtype=float),
        h_plus=np.asarray(h_plus, dtype=float),
        h_minus=np.asarray(h_minus, dtype=float),
        converged=converged,
        n_iter=n_iter,
        max_resid=max_resid,
        belief=belief,
        noise=noise,
        config=cfg,
        tail_plus_fit=tail_p,
        tail_minus_fit=tail_m,
    )


# ---------------------------------------------------------------------------
# Theoretical tail exponents (Lemma 3.3 / Thm. 5.1)
# ---------------------------------------------------------------------------


def theoretical_rho_sequence(
    kappa_end: float,
    n_informed: int,
    nu: float,
    n_periods: int,
) -> Array:
    """Endpoint tail exponents ``rho_t^+`` of Thm. 5.1 (ask side), by period.

    ``kappa_end`` is the posterior endpoint exponent of Assumption 3.3 /
    Remark 3.4: ``p_V(M - s) ~ c s^{kappa-1}`` (uniform: kappa=1; beta(a,b):
    kappa=b; density positive at M: kappa=1 with L=0). The recursion is the
    paper's ``alpha_t = kappa - (nu+1) sum_{s<t} 1/rho_s``,
    ``psi'_t = alpha_t / (alpha_t + 1)``,
    ``rho_t = (psi'_t - 1) / (1 - psi'_t / N_t)``
    (eqs. 5.4 and the Lemma 3.3 definition), equivalently
    ``rho_t = -1 / (1 + (N_t-1)/N_t * alpha_t)``. Fail-closed: kappa > 0,
    N > 1, nu > 0, n_periods >= 1.
    """
    kap = _pos_finite(kappa_end, "kappa_end")
    n = int(n_informed)
    if isinstance(n_informed, bool) or n < 2:
        raise ValueError("n_informed must be an integer > 1")
    nn = _pos_finite(nu, "nu")
    T = int(n_periods)
    if isinstance(n_periods, bool) or T < 1:
        raise ValueError("n_periods must be an integer >= 1")
    rho = np.empty(T, dtype=float)
    alpha = kap
    for t in range(T):
        rho[t] = -1.0 / (1.0 + ((n - 1.0) / n) * alpha)
        alpha = alpha - (nn + 1.0) / rho[t]
    return rho


def theoretical_informed_exponent(
    kappa_end: float,
    n_informed: int,
    nu: float,
    period: int = 1,
) -> float:
    """Cor. 5.1 exponent of ``P(X* >= y) = Pi+(F(y))`` at ``period``:

    ``(psi'_t / (1 - psi'_t)) * rho_t^+`` with
    ``psi'_t = alpha_t / (alpha_t + 1)``, i.e. ``alpha_t * rho_t^+`` —
    regularly-varying index of the informed-demand tail (the exponent that
    dominates the Student-t noise tail in the far tail).
    """
    rho_seq = theoretical_rho_sequence(kappa_end, n_informed, nu, period)
    kap = _pos_finite(kappa_end, "kappa_end")
    nn = _pos_finite(nu, "nu")
    alpha = kap
    for t in range(period):
        psi_prime = alpha / (alpha + 1.0)
        if t == period - 1:
            return float(psi_prime / (1.0 - psi_prime) * rho_seq[t])
        alpha = alpha - (nn + 1.0) / rho_seq[t]
    raise AssertionError("unreachable")


# ---------------------------------------------------------------------------
# Sequential learning run (eq. 3.3 posterior dynamics, Sec. 4)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LearningRun:
    """A T-period equilibrium learning session under a seeded noise path.

    Arrays are per trading period ``t = 1..T``: ``F_solutions[t]`` is the
    converged marginal-cost branch under the period-``t`` posterior;
    ``y[t]`` the realized aggregate flow ``X*_t(v0) + Z_t``; ``posterior``
    the belief entering the *next* solve. Consistency diagnostics (Sec. 4):
    posterior mean error ``|E_t[V] - v0|``, standard deviation, and the
    concentrated mass ``P(|V - v0| <= eps)``.
    """

    v0: float
    noise: NoiseSpec
    solutions: tuple[MarginalCostSolution, ...]
    posteriors: tuple[Belief, ...]  # posteriors[t] is the belief AFTER y[t]
    y: Array
    x_star: Array
    z: Array
    posterior_mean: Array
    posterior_std: Array
    mean_abs_error: Array
    mass_within_eps: Array
    spread: Array

    @property
    def n_periods(self) -> int:
        return int(self.y.size)

    def posterior_consistency_error(self) -> float:
        """``|E_T[V] - v0|`` at the terminal period (Sec. 4 diagnostic)."""
        return float(self.mean_abs_error[-1])

    def spread_persistence(self) -> float:
        """Spread ratio after the largest |Y| shock vs the period before it.

        The paper's persistent-spread diagnostic: heavy-tailed trades widen
        the adverse-selection spread and it decays only slowly. Returns
        ``spread[t_shock+1] / spread[t_shock]`` when the shock is not in the
        last period, else the ratio at the last period vs the first.
        """
        s = self.spread
        if s.size < 2:
            return 1.0
        t_shock = int(np.argmax(np.abs(self.y)))
        if t_shock + 1 < s.size:
            num, den = s[t_shock + 1], s[t_shock]
        else:
            num, den = s[-1], s[0]
        if den <= 0.0:
            raise ValueError("pre-shock spread is zero — persistence ratio undefined")
        return float(num / den)


def simulate_learning(
    prior: Belief,
    noise: NoiseSpec,
    config: SolverConfig | None,
    v0: float,
    n_periods: int,
    rng: np.random.Generator | int,
    eps: float | None = None,
) -> LearningRun:
    """Simulate T trading rounds of the sequential LOB (Secs. 3-4).

    Each period ``t``: solve ``F_t`` under the current posterior (warm-start
    from the previous branch — same discretization, faster convergence);
    informed demand ``X*_t(v) = F_t^{-1}(v)`` is evaluated on the whole
    v-grid for the belief update; the realized flow is
    ``Y_t = F_t^{-1}(v0) + Z_t`` with ``Z_t`` drawn from the seeded stream;
    beliefs update by the eq. (3.3) Student-t likelihood in log space.

    Deterministic: the same seed reproduces the run bit-identically.
    Fail-closed: ``v0`` must lie strictly inside ``(m, M)`` (the theory's
    interior ``Theta_eps`` — endpoint truth degenerates the likelihood);
    ``n_periods >= 1``; a non-converging period raises via the strict
    accessor rather than silently continuing.
    """
    cfg = config if config is not None else SolverConfig()
    vv0 = float(v0)
    if not math.isfinite(vv0) or not prior.m < vv0 < prior.M:
        raise ValueError(
            f"v0 must lie strictly inside the open support ({prior.m}, {prior.M}), got {v0!r}"
        )
    T = int(n_periods)
    if isinstance(n_periods, bool) or T < 1:
        raise ValueError("n_periods must be an integer >= 1")
    g = _resolve_rng(rng)
    ee = float(eps) if eps is not None else 0.1 * (prior.M - prior.m)
    if not math.isfinite(ee) or ee <= 0.0:
        raise ValueError("eps must be positive and finite")

    belief = prior
    F_prev: Array | None = None
    solutions: list[MarginalCostSolution] = []
    posteriors: list[Belief] = []
    ys = np.empty(T)
    xs = np.empty(T)
    zs = np.empty(T)
    pmean = np.empty(T)
    pstd = np.empty(T)
    perr = np.empty(T)
    pmass = np.empty(T)
    pspr = np.empty(T)

    for t in range(T):
        sol = solve_marginal_cost(belief, noise, cfg, F0=F_prev)
        sol.require_solution()  # fail-closed: never propagate an unconverged period
        x_of_v = sol._z_informed(belief.v)
        x0 = float(sol.informed_demand(np.asarray([vv0]))[0])
        z_t = float(noise.sample(g, 1)[0])
        y_t = x0 + z_t
        belief = belief.posterior(noise.logpdf(np.asarray(y_t - x_of_v, dtype=float)))
        solutions.append(sol)
        posteriors.append(belief)
        ys[t] = y_t
        xs[t] = x0
        zs[t] = z_t
        pmean[t] = belief.mean
        pstd[t] = belief.std
        perr[t] = abs(belief.mean - vv0)
        pmass[t] = belief.mass_within(vv0, ee)
        pspr[t] = sol.spread()
        F_prev = sol.F

    return LearningRun(
        v0=vv0,
        noise=noise,
        solutions=tuple(solutions),
        posteriors=tuple(posteriors),
        y=np.asarray(ys, dtype=float),
        x_star=np.asarray(xs, dtype=float),
        z=np.asarray(zs, dtype=float),
        posterior_mean=np.asarray(pmean, dtype=float),
        posterior_std=np.asarray(pstd, dtype=float),
        mean_abs_error=np.asarray(perr, dtype=float),
        mass_within_eps=np.asarray(pmass, dtype=float),
        spread=np.asarray(pspr, dtype=float),
    )


# ---------------------------------------------------------------------------
# Bench: flat SYNTHETIC diagnostic dict (seeded, ~seconds)
# ---------------------------------------------------------------------------


def liquidity_tail_bench(seed: int = 0, n_periods: int = 5) -> dict[str, float]:
    """Flat ``dict[str, float]`` of SYNTHETIC equilibrium diagnostics.

    Uniform prior on ``[0, 1]``, ``N = 2``, ``sigma = 2.0`` — large enough for
    an interior crossover depth (uninformed flow must dominate the marginal
    flow at moderate depths for the diagnostic to have dynamic range; with
    ``sigma << M - m`` informed demand dominates at every depth). Compares
    the heavy-tailed book (``nu = 3``) against the Gaussian benchmark on the
    same grid — the paper's core comparative static. Keys are all
    ``synthetic_*``; values are diagnostics, never market evidence.
    """
    g = np.random.default_rng(int(seed))
    cfg = SolverConfig(
        n_informed=2, x_max=10.0, n_x=101, z_max=30.0, n_z=301, tol=1e-9, max_iter=2000
    )
    prior = uniform_belief(0.0, 1.0, 201)
    v0 = 0.7

    out: dict[str, float] = {"synthetic_solver_revision": 1.0}
    sols: dict[str, MarginalCostSolution] = {}
    for tag, nz in (("t", student_t_noise(3.0, 2.0)), ("gauss", gaussian_noise(2.0))):
        sol = solve_marginal_cost(prior, nz, cfg)
        sols[tag] = sol
        out[f"synthetic_fixedpoint_max_resid_{tag}"] = float(sol.max_resid)
        out[f"synthetic_fixedpoint_iters_{tag}"] = float(sol.n_iter)
        out[f"synthetic_converged_{tag}"] = float(sol.converged)
        out[f"synthetic_spread_{tag}"] = sol.spread()
        out[f"synthetic_crossover_depth_{tag}"] = sol.crossover_depth()
        out[f"synthetic_h_deep_{tag}"] = float(sol.price(4.0))
        out[f"synthetic_tail_rho_hat_{tag}"] = sol.empirical_tail_exponent("ask")
    out["synthetic_crossover_ratio_t_over_gauss"] = (
        out["synthetic_crossover_depth_t"] / out["synthetic_crossover_depth_gauss"]
        if math.isfinite(out["synthetic_crossover_depth_t"])
        and math.isfinite(out["synthetic_crossover_depth_gauss"])
        and out["synthetic_crossover_depth_gauss"] > 0.0
        else float("inf")
    )
    out["synthetic_informed_dominance_size"] = sols["t"].crossover_depth()
    out["synthetic_informed_share_deep_t"] = float(sols["t"].informed_share(6.0))
    out["synthetic_informed_share_deep_gauss"] = float(sols["gauss"].informed_share(6.0))
    out["synthetic_far_tail_monotone_t"] = float(sols["t"].far_tail_monotone(0.7))
    out["synthetic_tail_rho_theory_t1"] = float(
        theoretical_rho_sequence(1.0, cfg.n_informed, 3.0, 1)[0]
    )
    out["synthetic_informed_exponent_theory_t1"] = theoretical_informed_exponent(
        1.0, cfg.n_informed, 3.0, 1
    )

    run = simulate_learning(prior, sols["t"].noise, cfg, v0, n_periods, g)
    out["synthetic_posterior_consistency_error"] = run.posterior_consistency_error()
    out["synthetic_posterior_std_initial"] = float(run.posterior_std[0])
    out["synthetic_posterior_std_final"] = float(run.posterior_std[-1])
    out["synthetic_posterior_mass_within_eps_final"] = float(run.mass_within_eps[-1])
    out["synthetic_spread_persistence_ratio"] = run.spread_persistence()
    out["synthetic_spread_first"] = float(run.spread[0])
    out["synthetic_spread_last"] = float(run.spread[-1])
    out["synthetic_n_periods"] = float(run.n_periods)
    return out


__all__ = [
    "Array",
    "Belief",
    "FixedPointError",
    "LearningRun",
    "MarginalCostSolution",
    "NoiseSpec",
    "PowerTail",
    "Side",
    "SolverConfig",
    "beta_belief",
    "gaussian_noise",
    "grid_belief",
    "liquidity_tail_bench",
    "simulate_learning",
    "solve_marginal_cost",
    "student_t_noise",
    "theoretical_informed_exponent",
    "theoretical_rho_sequence",
    "truncnorm_belief",
    "uniform_belief",
]
