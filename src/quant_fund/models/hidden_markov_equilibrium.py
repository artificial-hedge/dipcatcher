"""Equilibrium prices under hidden Markov fundamentals (two-state case).

Implements the two-state lane of Pagès, Possamaï & Rodriguez Polo (2026),
"Equilibrium prices under hidden Markov fundamentals", arXiv:2609.21684
[q-fin.MF] (citation verified against https://arxiv.org/abs/2609.21684 and the
full HTML text, fetched 2026-09-30).

Setting (Sec. 2): geometric dividends dD/D = g_t dt + sigma dW driven by an
unobserved two-state Markov chain g in {g1 = high, g2 = low} with strictly
positive transition intensities lam^{21} (low -> high) and lam^{12} (high ->
low).  A representative agent with Epstein-Zin preferences (risk aversion
gamma, elasticity psi, discount delta, theta = (1-gamma)/(1-1/psi)) observes
only dividends.  The belief p_t = P(g_t = g1 | F^D_t) follows the Wonham
equation (Prop. 2.1): dp = B(p) dt + chi(p) dWbar with

    B(p)   = lam^{21}(1-p) - lam^{12} p,
    chi(p) = p(1-p) Delta_g / sigma,   Delta_g = g1 - g2 >= 0,
    dWbar  = sigma^{-1} (dD/D - g(p) dt),  g(p) = g2 + p Delta_g.

Equilibrium (Sec. 3-4): the price-dividend ratio may a priori carry an
extra positive absolutely continuous factor h; inside the paper's class C
the equilibrium conditions force h constant, so Q_t = phi(p_t) with phi the
unique positive classical solution of the Markovian pricing equation
(4.1)-(4.2):

    -alpha (phi'' + (theta-1) phi'^2 / phi) - beta phi' + (c/theta) phi = 1,
    -lam^{21} phi'(0) + c(0) phi(0)/theta = 1,
    +lam^{12} phi'(1) + c(1) phi(1)/theta = 1,

where alpha = A^2/(2 sigma^2), A(p) = -p(1-p) Delta_g,
beta(p) = B(p) - (1-gamma) A(p), c(p) = kappa - (1-gamma) g(p) and
kappa = delta theta + gamma(1-gamma) sigma^2/2.  The semilinear transform
Phi = phi^theta, q = 1 - 1/theta gives (4.3)-(4.4):
-alpha Phi'' - beta Phi' + c Phi = theta Phi^q with matching boundary
relations.  Under Assumption 4.1 (c > 0 on [0,1]) and positive intensities
the positive solution exists, is unique, smooth on [0,1], analytic on
(0,1), and bounded by theta/c_max <= phi <= theta/c_min (Thm. 4.2-4.5).
theta = 1 gives the affine closed form phi(p) = a + b p (Prop. 4.11);
eta = (1-gamma) Delta_g > 0 makes phi strictly increasing (Thm. 4.10).

Options (Sec. 5, eta > 0 subregion): writing ell = log phi,
sigma_S(p) = sigma + ell'(p) chi(p) >= sigma, dividend yield
q_S = 1/phi, the marginal short rate (5.1)
r_f = g/psi + (theta-1)(d^2/(2 sigma^2) - d) + kappa/theta - gamma sigma^2
with d = -sigma chi ell', market price of belief risk
lambda(p) = gamma sigma + (1-theta) chi ell' (5.6), risk-neutral belief
drift Btilde = B - chi lambda (5.8).  European calls obey (5.10):
C_tau = L^Q C - r_f C on (X, p, tau) with the reduced one-sided-p boundary
equations of Thm. 5.4, and the leading conditional risk-neutral log-return
skewness is (Prop. 5.5)

    Skew^Q(tau) = 3 chi(p) sigma_S'(p) / sigma_S(p) * sqrt(tau) + o(sqrt(tau)).

Honesty: every numerical object here is either a solver diagnostic (BVP
residuals, the Thm. 4.2 uniform-bound check, factor flatness
sup|phi zeta - 1|, vol-of-belief shape) or a SYNTHETIC Monte-Carlo
correctness check (seeded simulation of the dividend/belief economy;
filtered-belief innovation calibration and PIT values are proper-score
diagnostics per the repo contract).  No Sharpe-family headline is produced;
option values are risk-neutral expectations, not P&L.
``simulate_dividends``, the ``mc_*`` helpers and ``bench_*`` are SYNTHETIC
correctness material, never market evidence.

Composition notes: ``hmm.discrete`` is Rabiner's finite-alphabet HMM
(discrete emissions, Viterbi/Baum-Welch) and ``models.io_hmm`` is its
input-driven discrete-time cousin; neither covers the continuous-time
Gaussian-innovation Wonham equation, so the filter is implemented here from
Prop. 2.1 via the exact two-state CTMC skeleton plus a Gaussian emission
update.  ``models.changepoint`` handles structural-break posteriors on a
level series, not Markov-switching drift.  The tridiagonal Newton BVP
solve, the 9-point Crank-Nicolson option grid and the shared-innovation
Q-sampler are new; no repo module owns them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import solve_banded
from scipy.sparse import csr_matrix, identity
from scipy.sparse.linalg import factorized
from scipy.special import ndtr

Array = NDArray[np.float64]

_EPS = 1e-15


def _finite_scalar(x: float, name: str) -> float:
    v = float(x)
    if not math.isfinite(v):
        raise ValueError(f"{name} must be finite")
    return v


def _positive_scalar(x: float, name: str) -> float:
    v = _finite_scalar(x, name)
    if v <= 0.0:
        raise ValueError(f"{name} must be > 0")
    return v


def _belief_vector(p: object) -> Array:
    v = np.asarray(p, dtype=float)
    if not bool(np.all(np.isfinite(v))):
        raise ValueError("p must be finite")
    return np.asarray(v, dtype=float)


# ---------------------------------------------------------------------------
# Economy primitives (Sec. 2, 4.1)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HiddenMarkovEconomy:
    """Two-state hidden-Markov Epstein-Zin economy (paper Sec. 2, 4.1).

    State 1 is the high-growth state, state 2 the low-growth state
    (``g_high`` >= ``g_low``; ``g_high == g_low`` is the degenerate-learning
    economy with chi == 0).  ``lam_up`` = lam^{21} (low -> high) and
    ``lam_down`` = lam^{12} (high -> low) are the strictly positive
    transition intensities of Thm. 4.2.  Preferences: ``gamma`` > 0 risk
    aversion, ``psi`` > 0 elasticity of intertemporal substitution
    (psi != 1, and the pair must give theta = (1-gamma)/(1-1/psi) > 0),
    ``delta`` > 0 subjective discount.
    """

    g_high: float
    g_low: float
    sigma: float
    lam_up: float
    lam_down: float
    gamma: float
    psi: float
    delta: float

    def __post_init__(self) -> None:
        g1 = _finite_scalar(self.g_high, "g_high")
        g2 = _finite_scalar(self.g_low, "g_low")
        if g1 < g2:
            raise ValueError("g_high must be >= g_low (state 1 is the high-growth state)")
        _positive_scalar(self.sigma, "sigma")
        _positive_scalar(self.lam_up, "lam_up")
        _positive_scalar(self.lam_down, "lam_down")
        _positive_scalar(self.gamma, "gamma")
        _positive_scalar(self.psi, "psi")
        _positive_scalar(self.delta, "delta")
        if float(self.psi) == 1.0:
            raise ValueError("psi = 1 is singular for theta = (1-gamma)/(1-1/psi)")
        if not self.theta > 0.0:
            raise ValueError(
                "theta = (1-gamma)/(1-1/psi) must be > 0: gamma and psi must lie "
                "on the same side of 1 (paper assumes theta > 0)"
            )

    @property
    def delta_g(self) -> float:
        """Delta_g = g1 - g2 >= 0."""
        return float(self.g_high - self.g_low)

    @property
    def theta(self) -> float:
        """theta = (1-gamma)/(1-psi^{-1})."""
        return float((1.0 - self.gamma) / (1.0 - 1.0 / self.psi))

    @property
    def kappa(self) -> float:
        """kappa = delta*theta + gamma(1-gamma) sigma^2 / 2."""
        return float(
            self.delta * self.theta + self.gamma * (1.0 - self.gamma) * self.sigma**2 / 2.0
        )

    @property
    def eta(self) -> float:
        """eta = (1-gamma) Delta_g; the Sec. 5 subregion is eta > 0."""
        return float((1.0 - self.gamma) * self.delta_g)

    @property
    def q(self) -> float:
        """q = 1 - 1/theta < 1, the sublinear exponent of (4.3)."""
        return float(1.0 - 1.0 / self.theta)

    @property
    def stationary_high(self) -> float:
        """Stationary P(high) = lam^{21} / (lam^{21} + lam^{12})."""
        return float(self.lam_up / (self.lam_up + self.lam_down))

    def transition_matrix(self, dt: float) -> Array:
        """P(X_{t+dt} = j | X_t = i), states ordered (high, low); exact CTMC.

        With lam_sum = lam_up + lam_down, stationary masses
        (pi_h, pi_l) = (lam_up, lam_down)/lam_sum and rho = e^{-lam_sum dt}:
        T = [[pi_h + pi_l rho, pi_l (1-rho)], [pi_h (1-rho), pi_l + pi_h rho]].
        """
        h = _positive_scalar(dt, "dt")
        lam = self.lam_up + self.lam_down
        rho = math.exp(-lam * h)
        pi_h = self.lam_up / lam
        pi_l = self.lam_down / lam
        return np.asarray(
            [
                [pi_h + pi_l * rho, pi_l * (1.0 - rho)],
                [pi_h * (1.0 - rho), pi_l + pi_h * rho],
            ],
            dtype=float,
        )

    def growth(self, p: object) -> Array:
        """g(p) = g2 + p Delta_g — the conditional-mean dividend drift."""
        return np.asarray(self.g_low + _belief_vector(p) * self.delta_g, dtype=float)

    def costate(self, p: object) -> Array:
        """c(p) = kappa - (1-gamma) g(p); Assumption 4.1 needs c > 0 on [0,1]."""
        return np.asarray(self.kappa - (1.0 - self.gamma) * self.growth(p), dtype=float)

    @property
    def c_min(self) -> float:
        """min c on [0,1] (c is affine, so the bound is an endpoint)."""
        return float(min(self.costate(0.0), self.costate(1.0)))

    @property
    def c_max(self) -> float:
        """max c on [0,1]."""
        return float(max(self.costate(0.0), self.costate(1.0)))

    @property
    def positivity_holds(self) -> bool:
        """Assumption 4.1: c(p) > 0 for all p in [0,1]."""
        return bool(self.c_min > 0.0)


def calibrate_figure_economy(theta: float = 0.25) -> HiddenMarkovEconomy:
    """Sec. 4.4 primitive calibration: g1 = 0.25, g2 = 0, gamma = 0.8,
    sigma = 1.25, lam^{12} = lam^{21} = 0.04, delta = 0.025, with
    psi(theta) = (1 - 0.2/theta)^{-1} so theta is the free dial."""
    th = _positive_scalar(theta, "theta")
    psi = 1.0 / (1.0 - 0.2 / th)
    return HiddenMarkovEconomy(
        g_high=0.25,
        g_low=0.0,
        sigma=1.25,
        lam_up=0.04,
        lam_down=0.04,
        gamma=0.8,
        psi=psi,
        delta=0.025,
    )


def calibrate_option_economy() -> HiddenMarkovEconomy:
    """Sec. 5.5 option illustration: g1 = 0.05, g2 = -0.05, sigma = 0.04,
    delta = 0.06, lam^{12} = lam^{21} = 0.05, gamma = 0.8, theta = 0.25,
    psi = 5.  The paper reports min c = 0.005128, sigma_S max ~= 35.6% near
    p* ~= 0.486 and a marginal-short-rate range of 1.17%-6.92%."""
    return HiddenMarkovEconomy(
        g_high=0.05,
        g_low=-0.05,
        sigma=0.04,
        lam_up=0.05,
        lam_down=0.05,
        gamma=0.8,
        psi=5.0,
        delta=0.06,
    )


# ---------------------------------------------------------------------------
# Belief dynamics coefficients (Prop. 2.1, Sec. 4.1, 5.2)
# ---------------------------------------------------------------------------


def belief_drift(p: object, econ: HiddenMarkovEconomy) -> Array:
    """B(p) = lam^{21}(1-p) - lam^{12} p (physical Wonham drift)."""
    pp = _belief_vector(p)
    return np.asarray(econ.lam_up * (1.0 - pp) - econ.lam_down * pp, dtype=float)


def belief_vol(p: object, econ: HiddenMarkovEconomy) -> Array:
    """chi(p) = p(1-p) Delta_g / sigma: vol of belief; zero at p in {0,1}."""
    pp = _belief_vector(p)
    return np.asarray(pp * (1.0 - pp) * econ.delta_g / econ.sigma, dtype=float)


def _coeff_A(p: Array, econ: HiddenMarkovEconomy) -> Array:
    """A(p) = -p(1-p) Delta_g = -sigma chi(p) (paper's simplex A)."""
    return np.asarray(-p * (1.0 - p) * econ.delta_g, dtype=float)


def _coeff_alpha(p: Array, econ: HiddenMarkovEconomy) -> Array:
    """alpha(p) = A(p)^2 / (2 sigma^2) — degenerate diffusion weight."""
    a = _coeff_A(p, econ)
    return np.asarray(a * a / (2.0 * econ.sigma**2), dtype=float)


def _coeff_beta(p: Array, econ: HiddenMarkovEconomy) -> Array:
    """beta(p) = B(p) - (1-gamma) A(p) = lam^{21}(1-p) - lam^{12}p + eta p(1-p)."""
    return np.asarray(belief_drift(p, econ) - (1.0 - econ.gamma) * _coeff_A(p, econ), dtype=float)


# ---------------------------------------------------------------------------
# Wonham belief filter on a dividend stream (Prop. 2.1)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BeliefPath:
    """Wonham-filtered posterior path.

    ``filtered[t]`` = P(g_{t+1} = high | dividends up to index t+1);
    ``predicted[t]`` is the pre-observation belief after the CTMC predict
    step; ``innovations[t]`` is the predictive-standardized residual of the
    log-dividend increment (unit variance under the true model — a
    proper-score calibration object); ``pit[t]`` is the predictive-mixture
    PIT value; ``log_likelihood`` is the cumulative log predictive score
    (the HMM likelihood — itself a proper score).
    """

    filtered: Array
    predicted: Array
    innovations: Array
    pit: Array
    log_likelihood: float
    dt: float


def _as_dividend_levels(dividends: object, min_len: int = 2) -> Array:
    d = np.asarray(dividends, dtype=float).reshape(-1)
    if d.size < min_len:
        raise ValueError(f"dividends must contain >= {min_len} observations")
    if not bool(np.all(np.isfinite(d))):
        raise ValueError("dividends must be finite")
    if bool(np.any(d <= 0.0)):
        raise ValueError("dividends must be strictly positive (geometric dividend)")
    return d


def wonham_filter(
    dividends: object,
    dt: float,
    econ: HiddenMarkovEconomy,
    p0: float | None = None,
) -> BeliefPath:
    """Two-state Wonham filter (paper Prop. 2.1) on observed dividend levels.

    Discretization: predict through the exact CTMC skeleton T(dt), then a
    Gaussian emission update for the log-dividend increment
    Delta log D ~ N((g_i - sigma^2/2) dt, sigma^2 dt) under state i.  This
    is the standard strong discretization of the Wonham (finite-state
    Kushner-Stratonovich) equation; it keeps the posterior exactly inside
    (0,1).  ``p0`` defaults to the stationary belief.  Fail-closed on
    nonpositive/nonfinite dividends, dt <= 0, or p0 outside (0,1).
    """
    d = _as_dividend_levels(dividends)
    h = _positive_scalar(dt, "dt")
    pp0 = econ.stationary_high if p0 is None else _finite_scalar(p0, "p0")
    if not 0.0 < pp0 < 1.0:
        raise ValueError("p0 must lie in (0, 1)")

    trans = econ.transition_matrix(h)  # rows (from high, from low)
    inc = np.diff(np.log(d))
    n = inc.size
    var_d = econ.sigma**2 * h
    sd_d = math.sqrt(var_d)
    mu_h = (econ.g_high - 0.5 * econ.sigma**2) * h
    mu_l = (econ.g_low - 0.5 * econ.sigma**2) * h

    filtered = np.empty(n)
    predicted = np.empty(n)
    innovations = np.empty(n)
    pit = np.empty(n)
    loglik = 0.0
    post = pp0
    for t in range(n):
        pred_h = post * trans[0, 0] + (1.0 - post) * trans[1, 0]
        pred_l = 1.0 - pred_h
        predicted[t] = pred_h
        z_h = (inc[t] - mu_h) / sd_d
        z_l = (inc[t] - mu_l) / sd_d
        like_h = math.exp(-0.5 * z_h * z_h) / sd_d
        like_l = math.exp(-0.5 * z_l * z_l) / sd_d
        w_h = pred_h * like_h
        w_l = pred_l * like_l
        tot = w_h + w_l
        if tot <= 0.0 or not math.isfinite(tot):
            # probability-zero guard: keep the prior prediction rather than
            # emitting a NaN belief (never reached for finite inputs)
            post = pred_h
        else:
            post = w_h / tot
        loglik += math.log(max(tot, 1e-300))
        m_pred = pred_h * mu_h + pred_l * mu_l
        v_pred = pred_h * pred_l * (mu_h - mu_l) ** 2 + var_d
        innovations[t] = (inc[t] - m_pred) / math.sqrt(v_pred)
        pit[t] = pred_h * float(ndtr(z_h)) + pred_l * float(ndtr(z_l))
        filtered[t] = post
    return BeliefPath(
        filtered=filtered,
        predicted=predicted,
        innovations=innovations,
        pit=pit,
        log_likelihood=loglik,
        dt=h,
    )


@dataclass(frozen=True)
class SyntheticDividendPath:
    """SYNTHETIC joint path of the hidden state and the dividend stream."""

    states: NDArray[np.int64]  # int states on the observation grid: 1 = high, 0 = low
    levels: Array  # D_t with D_0 = 1
    log_increments: Array
    dt: float


def simulate_dividends(
    econ: HiddenMarkovEconomy,
    n_steps: int,
    dt: float,
    seed: int,
    x0: int | float | None = None,
) -> SyntheticDividendPath:
    """SYNTHETIC exact-skeleton Monte-Carlo path (correctness material only).

    The hidden chain is sampled exactly at grid times from the CTMC
    transition matrix (no Euler error), and log-dividend increments use the
    exact Gaussian law conditional on the state over each step.
    ``x0``: 0/1 for a fixed initial state or a probability for a seeded
    draw; None draws from the stationary law.
    """
    if isinstance(n_steps, bool) or int(n_steps) < 2:
        raise ValueError("n_steps must be an integer >= 2")
    n = int(n_steps)
    h = _positive_scalar(dt, "dt")
    rng = np.random.default_rng(int(seed))
    trans = econ.transition_matrix(h)
    if x0 is None:
        init: float = float(rng.random() < econ.stationary_high)
    else:
        xx = float(x0)
        if xx in (0.0, 1.0):
            init = xx
        elif 0.0 < xx < 1.0:
            init = float(rng.random() < xx)
        else:
            raise ValueError("x0 must be 0 (low), 1 (high), an initial probability, or None")
    states = np.empty(n + 1, dtype=np.int64)
    states[0] = int(init)
    flips = rng.random(n)
    z = rng.standard_normal(n)
    inc = np.empty(n)
    for t in range(n):
        cur = int(states[t])
        if flips[t] < trans[cur, 1 - cur]:
            states[t + 1] = 1 - cur
        else:
            states[t + 1] = cur
        g = econ.g_high if cur == 1 else econ.g_low
        inc[t] = (g - 0.5 * econ.sigma**2) * h + econ.sigma * math.sqrt(h) * z[t]
    levels = np.exp(np.concatenate([[0.0], np.cumsum(inc)]))
    return SyntheticDividendPath(states=states, levels=levels, log_increments=inc, dt=h)


# ---------------------------------------------------------------------------
# Two-state Markovian pricing equation (Sec. 4) — Newton on the Phi-BVP
# ---------------------------------------------------------------------------


def _bvp_residual_and_jacobian(
    phi_pow: Array,
    p: Array,
    econ: HiddenMarkovEconomy,
) -> tuple[Array, Array, Array, Array]:
    """Residual F(Phi) of (4.3)-(4.4) and its tridiagonal Jacobian.

    Interior row i: -alpha_i D2Phi - beta_i D1Phi + c_i Phi_i - theta Phi_i^q.
    Row 0: -lam^{21} (Phi_1 - Phi_0)/h + c_0 Phi_0 - theta Phi_0^q;
    row n-1: +lam^{12} (Phi_{n-1} - Phi_{n-2})/h + c_1 Phi - theta Phi^q.
    Returns (F, lower, diag, upper) where upper[i] couples row i to column
    i+1 and lower[i] couples row i+1 to column i (solve_banded layout).
    """
    n = p.size
    h = 1.0 / (n - 1)
    theta, qq = econ.theta, econ.q
    alpha = _coeff_alpha(p, econ)
    beta = _coeff_beta(p, econ)
    c = econ.costate(p)
    rhs = theta * phi_pow**qq
    rhs_p = theta * qq * phi_pow ** (qq - 1.0)

    f = np.empty(n)
    lower = np.zeros(n - 1)
    diag = np.empty(n)
    upper = np.zeros(n - 1)

    d2 = (phi_pow[:-2] - 2.0 * phi_pow[1:-1] + phi_pow[2:]) / h**2
    d1 = (phi_pow[2:] - phi_pow[:-2]) / (2.0 * h)
    a_i = alpha[1:-1]
    b_i = beta[1:-1]
    f[1:-1] = -a_i * d2 - b_i * d1 + c[1:-1] * phi_pow[1:-1] - rhs[1:-1]
    # dF_i/dPhi_{i-1} = -a/h^2 + b/(2h) -> lower[i-1]; row range i = 1..n-2
    lower[: n - 2] = -a_i / h**2 + b_i / (2.0 * h)
    # dF_i/dPhi_{i+1} = -a/h^2 - b/(2h) -> upper[i]; i = 1..n-2
    upper[1 : n - 1] = -a_i / h**2 - b_i / (2.0 * h)
    diag[1:-1] = 2.0 * a_i / h**2 + c[1:-1] - rhs_p[1:-1]

    # boundary row 0: -lam^{21} (Phi_1 - Phi_0)/h + c0 Phi0 - theta Phi0^q
    f[0] = -econ.lam_up * (phi_pow[1] - phi_pow[0]) / h + c[0] * phi_pow[0] - rhs[0]
    diag[0] = econ.lam_up / h + c[0] - rhs_p[0]
    upper[0] = -econ.lam_up / h

    # boundary row n-1: +lam^{12} (Phi_{n-1} - Phi_{n-2})/h + c1 Phi - theta Phi^q
    f[-1] = econ.lam_down * (phi_pow[-1] - phi_pow[-2]) / h + c[-1] * phi_pow[-1] - rhs[-1]
    diag[-1] = econ.lam_down / h + c[-1] - rhs_p[-1]
    lower[n - 2] = -econ.lam_down / h

    return f, lower, diag, upper


@dataclass(frozen=True)
class BeliefMarkovianSolution:
    """Solved belief-Markovian price map phi(p) = price-dividend ratio.

    ``Phi = phi^theta`` is the Newton solution of (4.3)-(4.4) on the uniform
    grid ``p``.  ``ell_p``/``ell_pp`` are the first two derivatives of
    ell = log phi; interior values come from centered differences and the
    endpoints from the exact (4.4)/(4.7) derivative relations.  Coefficient
    methods evaluate at arbitrary beliefs by linear interpolation on the
    fine solution grid.
    """

    econ: HiddenMarkovEconomy
    p: Array
    phi: Array
    phi_p: Array
    phi_pp: Array
    ell_p: Array
    ell_pp: Array
    Phi: Array
    n_iter: int
    residual_inf: float

    def _grid(self, values: Array, x: object) -> Array:
        xx = _belief_vector(x)
        return np.asarray(np.interp(xx, self.p, values), dtype=float)

    # -- price map ---------------------------------------------------------
    def price_dividend_ratio(self, p: object) -> Array:
        """phi(p): the equilibrium price-dividend ratio."""
        return self._grid(self.phi, p)

    def dividend_yield(self, p: object) -> Array:
        """q_S(p) = 1/phi(p): the instantaneous dividend yield."""
        return np.asarray(1.0 / self._grid(self.phi, p), dtype=float)

    # -- physical stock dynamics (Lemma 5.1) --------------------------------
    def stock_vol(self, p: object) -> Array:
        """sigma_S(p) = sigma + ell'(p) chi(p); >= sigma under eta > 0.

        chi vanishes at p in {0,1} so sigma_S(0) = sigma_S(1) = sigma: the
        pure-Brownian endpoint recovery.  (Sec. 5 assumes eta > 0; for
        eta <= 0, ell' may turn negative and sigma_S < sigma is possible.)
        """
        pp = _belief_vector(p)
        ell = self._grid(self.ell_p, pp)
        return np.asarray(self.econ.sigma + ell * belief_vol(pp, self.econ), dtype=float)

    def stock_vol_prime(self, p: object) -> Array:
        """sigma_S'(p) = chi'(p) ell'(p) + chi(p) ell''(p)."""
        pp = _belief_vector(p)
        chi_p = (1.0 - 2.0 * pp) * self.econ.delta_g / self.econ.sigma
        ell = self._grid(self.ell_p, pp)
        ell2 = self._grid(self.ell_pp, pp)
        return np.asarray(chi_p * ell + belief_vol(pp, self.econ) * ell2, dtype=float)

    def stock_drift(self, p: object) -> Array:
        """mu_S(p) = g + ell' B + ell'' chi^2/2 + (sigma_S^2 - sigma^2)/2 (5.5)."""
        pp = _belief_vector(p)
        ell = self._grid(self.ell_p, pp)
        ell2 = self._grid(self.ell_pp, pp)
        chi = belief_vol(pp, self.econ)
        sig_s = self.stock_vol(pp)
        return np.asarray(
            self.econ.growth(pp)
            + ell * belief_drift(pp, self.econ)
            + 0.5 * ell2 * chi * chi
            + 0.5 * (sig_s * sig_s - self.econ.sigma**2),
            dtype=float,
        )

    # -- risk-neutral objects (Sec. 5.2; paper's eta > 0 subregion) ---------
    def _require_eta_positive(self) -> None:
        if not self.econ.eta > 0.0:
            raise ValueError(
                "risk-neutral and option objects belong to the paper's eta > 0 "
                "subregion (eta = (1-gamma) Delta_g)"
            )

    def _d_coeff(self, p: object) -> Array:
        """d(p) = -sigma chi(p) ell'(p): the auxiliary coefficient of (5.1)."""
        pp = _belief_vector(p)
        return np.asarray(
            -self.econ.sigma * belief_vol(pp, self.econ) * self._grid(self.ell_p, pp),
            dtype=float,
        )

    def marginal_short_rate(self, p: object) -> Array:
        """r_f(p): the marginal short rate (5.1), the Sec. 5 selection.

        r_f = g/psi + (theta-1)(d^2/(2 sigma^2) - d) + kappa/theta - gamma sigma^2.
        """
        self._require_eta_positive()
        pp = _belief_vector(p)
        e = self.econ
        d = self._d_coeff(pp)
        return np.asarray(
            e.growth(pp) / e.psi
            + (e.theta - 1.0) * (d * d / (2.0 * e.sigma**2) - d)
            + e.kappa / e.theta
            - e.gamma * e.sigma**2,
            dtype=float,
        )

    def market_price_of_risk(self, p: object) -> Array:
        """lambda(p) = gamma sigma + (1-theta) chi ell' (eq. 5.6)."""
        self._require_eta_positive()
        pp = _belief_vector(p)
        return np.asarray(
            self.econ.gamma * self.econ.sigma
            + (1.0 - self.econ.theta) * belief_vol(pp, self.econ) * self._grid(self.ell_p, pp),
            dtype=float,
        )

    def rn_belief_drift(self, p: object) -> Array:
        """Btilde(p) = B - chi lambda = B - gamma sigma chi - (1-theta) chi^2 ell' (5.8)."""
        self._require_eta_positive()
        pp = _belief_vector(p)
        chi = belief_vol(pp, self.econ)
        return np.asarray(
            belief_drift(pp, self.econ) - chi * self.market_price_of_risk(pp), dtype=float
        )

    def log_stock_drift_q(self, p: object) -> Array:
        """b_X(p) = r_f - q_S - sigma_S^2/2 (risk-neutral log-stock drift)."""
        pp = _belief_vector(p)
        sig_s = self.stock_vol(pp)
        return np.asarray(
            self.marginal_short_rate(pp) - self.dividend_yield(pp) - 0.5 * sig_s * sig_s,
            dtype=float,
        )

    # -- short-maturity skewness (Prop. 5.5) --------------------------------
    def skewness_coefficient(self, p: object) -> Array:
        """3 chi(p) sigma_S'(p) / sigma_S(p): the sqrt(tau) coefficient."""
        self._require_eta_positive()
        pp = _belief_vector(p)
        return np.asarray(
            3.0 * belief_vol(pp, self.econ) * self.stock_vol_prime(pp) / self.stock_vol(pp),
            dtype=float,
        )

    def leading_skewness(self, p: object, tau: float) -> Array:
        """Skew^Q(tau) ~ skewness_coefficient(p) sqrt(tau) (Prop. 5.5)."""
        t = _positive_scalar(tau, "tau")
        return np.asarray(self.skewness_coefficient(p) * math.sqrt(t), dtype=float)


def solve_price_dividend_ratio(
    econ: HiddenMarkovEconomy,
    n_grid: int = 1601,
    tol: float = 1e-9,
    max_iter: int = 200,
) -> BeliefMarkovianSolution:
    """Solve the two-state pricing BVP (4.3)-(4.4) for Phi = phi^theta.

    Damped Newton on the centered-finite-difference semilinear system with
    the exact state-constraint boundary rows; positivity of the iterate is
    enforced by step halving.  Deterministic.  theta = 1 is linear and
    converges in a single Newton step.  The attained inf-norm residual of
    the discrete system is reported in ``residual_inf`` (the FD-conditioned
    floor sits around 1e-10 on a 1601-point grid; ``tol`` is the target).

    Fail-closed: Assumption 4.1 (c > 0 on [0,1]), n_grid >= 33, tol > 0,
    and a RuntimeError when the residual cannot be driven below ``tol``.
    """
    if not econ.positivity_holds:
        raise ValueError(
            "Assumption 4.1 fails: c(p) = kappa - (1-gamma) g(p) must be > 0 on "
            f"[0,1] (c_min = {econ.c_min:.6g})"
        )
    if isinstance(n_grid, bool) or int(n_grid) < 33:
        raise ValueError("n_grid must be an integer >= 33")
    n = int(n_grid)
    ttol = _positive_scalar(tol, "tol")
    if int(max_iter) < 1:
        raise ValueError("max_iter must be >= 1")

    p = np.linspace(0.0, 1.0, n)
    h = 1.0 / (n - 1)
    c_bar = float(np.mean(econ.costate(p)))
    phi_pow = np.full(n, (econ.theta / c_bar) ** econ.theta)  # inside Thm. 4.2 bounds

    f, lower, diag, upper = _bvp_residual_and_jacobian(phi_pow, p, econ)
    res = float(np.max(np.abs(f)))
    it = 0
    converged = res <= ttol
    while not converged:
        it += 1
        if it > int(max_iter):
            raise RuntimeError(
                f"Newton solve of (4.3)-(4.4) did not converge in {max_iter} "
                f"iterations (inf-norm {res:.3e})"
            )
        ab = np.zeros((3, n))
        ab[0, 1:] = upper
        ab[1, :] = diag
        ab[2, :-1] = lower
        step = solve_banded((1, 1), ab, f, overwrite_ab=True, check_finite=False)
        omega = 1.0
        accepted = False
        for _ in range(60):
            cand = phi_pow - omega * step
            if float(np.min(cand)) > 0.0:
                f_c, lo_c, di_c, up_c = _bvp_residual_and_jacobian(cand, p, econ)
                res_c = float(np.max(np.abs(f_c)))
                if res_c < res:
                    phi_pow, f, lower, diag, upper = cand, f_c, lo_c, di_c, up_c
                    res = res_c
                    accepted = True
                    break
            omega *= 0.5
        if not accepted:
            raise RuntimeError(
                f"Newton solve of (4.3)-(4.4) failed to reduce the residual "
                f"(inf-norm {res:.3e}) at iteration {it}"
            )
        converged = res <= ttol

    # Derivative recovery. Interior Phi' by centered FD; endpoint Phi'
    # exactly from the (4.4) boundary relations; interior Phi'' by centered
    # FD; endpoint Phi'' from the (4.7) j=1 relations
    # lam^{21} Phi''(0) = E_0'(0) - beta'(0) Phi'(0) and
    # lam^{12} Phi''(1) = -(E_0'(1) - beta'(1) Phi'(1)),
    # with E_0 = c Phi - theta Phi^q.
    rhs_pow = econ.theta * phi_pow**econ.q
    ph_p = np.empty(n)
    ph_p[1:-1] = (phi_pow[2:] - phi_pow[:-2]) / (2.0 * h)
    c = econ.costate(p)
    c_p = -(1.0 - econ.gamma) * econ.delta_g  # c'(p), constant
    beta_p = -(econ.lam_up + econ.lam_down) + econ.eta * (1.0 - 2.0 * p)
    ph_p[0] = (c[0] * phi_pow[0] - rhs_pow[0]) / econ.lam_up
    ph_p[-1] = (rhs_pow[-1] - c[-1] * phi_pow[-1]) / econ.lam_down
    ph_pp = np.empty(n)
    ph_pp[1:-1] = (phi_pow[:-2] - 2.0 * phi_pow[1:-1] + phi_pow[2:]) / h**2
    e0_p = c_p * phi_pow + c * ph_p - econ.theta * econ.q * phi_pow ** (econ.q - 1.0) * ph_p
    ph_pp[0] = (e0_p[0] - beta_p[0] * ph_p[0]) / econ.lam_up
    ph_pp[-1] = -(e0_p[-1] - beta_p[-1] * ph_p[-1]) / econ.lam_down

    theta = econ.theta
    ell_p = ph_p / (theta * phi_pow)
    ell_pp = ph_pp / (theta * phi_pow) - ph_p * ph_p / (theta * phi_pow * phi_pow)
    sol_phi = phi_pow ** (1.0 / theta)
    sol_phi_p = sol_phi * ell_p
    sol_phi_pp = sol_phi * (ell_pp + ell_p * ell_p)

    return BeliefMarkovianSolution(
        econ=econ,
        p=p,
        phi=sol_phi,
        phi_p=sol_phi_p,
        phi_pp=sol_phi_pp,
        ell_p=ell_p,
        ell_pp=ell_pp,
        Phi=phi_pow,
        n_iter=it,
        residual_inf=res,
    )


def closed_form_phi_theta1(econ: HiddenMarkovEconomy, p: object) -> Array:
    """Prop. 4.11 exact affine solution for theta = 1: phi(p) = a + b p.

    b = (c(0)-c(1)) / (c(0)c(1) + c(0) lam^{12} + c(1) lam^{21});
    a = (1 + lam^{21} b) / c(0).  Fail-closed unless theta == 1 and
    Assumption 4.1 holds.
    """
    if not math.isclose(econ.theta, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("closed_form_phi_theta1 requires theta == 1")
    if not econ.positivity_holds:
        raise ValueError("Assumption 4.1 fails: c(p) must be > 0 on [0,1]")
    c0 = float(econ.costate(0.0))
    c1 = float(econ.costate(1.0))
    b = (c0 - c1) / (c0 * c1 + c0 * econ.lam_down + c1 * econ.lam_up)
    a = (1.0 + econ.lam_up * b) / c0
    pp = _belief_vector(p)
    return np.asarray(a + b * pp, dtype=float)


# ---------------------------------------------------------------------------
# Diagnostics: ODE residual, factor flatness, vol-of-belief shape
# ---------------------------------------------------------------------------


def pricing_diagnostics(sol: BeliefMarkovianSolution, n_check: int = 201) -> dict[str, float]:
    """Solver-quality diagnostics for the belief-Markovian solution.

    - ``ode_residual_max``: sup over interior check points of
      |(4.1) LHS - 1| in the phi form, evaluated on an independent
      ``n_check``-point grid from interpolated phi'/phi''.
    - ``bc_residual_max``: the two state-constraint residuals of (4.2),
      evaluated on the solver grid's exact endpoint derivatives.
    - ``factor_flatness``: sup |phi zeta - 1| where zeta(p) is the Sec. 3
      normalized-identity coefficient
      zeta = kappa/theta - (1/(2 sigma^2)) (A^2 phi''/phi
      + (theta-1) (A phi'/phi)^2) - (b - (1-gamma) A) phi'/phi
      - (1-1/psi) g(p).  The pricing equation is phi zeta = 1, so this
      measures numerically how far the computed price map is from forcing
      the extra valuation factor h to be constant (belief-Markovian prices).
    - ``phi_lower_bound``/``phi_upper_bound``: theta/c_max, theta/c_min —
      the Thm. 4.2 uniform bounds — beside the attained phi range.
    """
    econ = sol.econ
    pp = np.linspace(0.0, 1.0, max(int(n_check), 33))
    phi = sol.price_dividend_ratio(pp)
    phi_p = np.asarray(np.interp(pp, sol.p, sol.phi_p), dtype=float)
    phi_pp = np.asarray(np.interp(pp, sol.p, sol.phi_pp), dtype=float)
    alpha = _coeff_alpha(pp, econ)
    beta = _coeff_beta(pp, econ)
    c = econ.costate(pp)
    theta = econ.theta

    interior = (pp > 0.0) & (pp < 1.0)
    f_int = np.abs(
        -alpha[interior] * (phi_pp[interior] + (theta - 1.0) * phi_p[interior] ** 2 / phi[interior])
        - beta[interior] * phi_p[interior]
        + c[interior] / theta * phi[interior]
        - 1.0
    )
    bc0 = -econ.lam_up * float(sol.phi_p[0]) + float(c[0]) * float(sol.phi[0]) / theta - 1.0
    bc1 = +econ.lam_down * float(sol.phi_p[-1]) + float(c[-1]) * float(sol.phi[-1]) / theta - 1.0

    a = _coeff_A(pp, econ)
    b = belief_drift(pp, econ)
    zeta = (
        econ.kappa / theta
        - (a * a * phi_pp / phi + (theta - 1.0) * (a * phi_p / phi) ** 2) / (2.0 * econ.sigma**2)
        - (b - (1.0 - econ.gamma) * a) * phi_p / phi
        - (1.0 - 1.0 / econ.psi) * econ.growth(pp)
    )
    flat = np.abs(phi * zeta - 1.0)

    chi = belief_vol(pp, econ)
    sig_s = sol.stock_vol(pp)
    i_star = int(np.argmax(sig_s))
    return {
        "ode_residual_max": float(np.max(f_int)),
        "bc_residual_max": float(max(abs(bc0), abs(bc1))),
        "factor_flatness": float(np.max(flat)),
        "phi_min": float(np.min(phi)),
        "phi_max": float(np.max(phi)),
        "phi_lower_bound": float(theta / econ.c_max),
        "phi_upper_bound": float(theta / econ.c_min),
        "sigmaS_min": float(np.min(sig_s)),
        "sigmaS_max": float(np.max(sig_s)),
        "p_star": float(pp[i_star]),
        "chi_max": float(np.max(chi)),
        "sigmaS_endpoint0": float(sig_s[0]),
        "sigmaS_endpoint1": float(sig_s[-1]),
        "n_iter": float(sol.n_iter),
        "residual_inf": float(sol.residual_inf),
    }


# ---------------------------------------------------------------------------
# Risk-neutral Monte-Carlo (Prop. 5.3 dynamics; Sec. 5.5 scheme)
# ---------------------------------------------------------------------------


def _mc_paths_q(
    sol: BeliefMarkovianSolution,
    p0: float,
    tau: float,
    z: Array,
    x0: float,
) -> tuple[Array, Array]:
    """Shared-innovation risk-neutral paths of (X, p).

    ``z`` is an (n_steps, n_paths) standard-normal matrix — the single
    Brownian increment driving BOTH the stock and the belief (the model's
    common innovation, which is what generates the return skew).  Scheme
    follows Sec. 5.5: log-Euler for the stock, Euler on logit(p) to keep
    beliefs inside (0,1).  Returns (log_returns = X_tau - x0, pathwise
    discount factors exp(-int r_f ds) by left sums).  Deterministic given
    ``z``.
    """
    tt = _positive_scalar(tau, "tau")
    if z.ndim != 2 or z.shape[0] < 1 or z.shape[1] < 1:
        raise ValueError("z must be a non-empty (n_steps, n_paths) array")
    n_t = z.shape[0]
    h = tt / n_t
    sq = math.sqrt(h)
    x = np.full(z.shape[1], float(x0))
    y = np.full(z.shape[1], math.log(p0 / (1.0 - p0)))
    disc = np.ones(z.shape[1])
    for t in range(n_t):
        p = 1.0 / (1.0 + np.exp(-y))
        sig_s = sol.stock_vol(p)
        chi = belief_vol(p, sol.econ)
        btil = sol.rn_belief_drift(p)
        rf = sol.marginal_short_rate(p)
        bx = sol.log_stock_drift_q(p)
        disc *= np.exp(-rf * h)
        x += bx * h + sig_s * sq * z[t]
        # Ito drift of y = logit(p): btil/(p(1-p)) + (2p-1) chi^2/(2 p^2 (1-p)^2);
        # diffusion chi/(p(1-p)) — driven by the SAME z as the stock.
        j = p * (1.0 - p)
        y += (btil / j + (2.0 * p - 1.0) * chi * chi / (2.0 * j * j)) * h + (chi / j) * sq * z[t]
    return x - x0, disc


def mc_log_returns_q(
    sol: BeliefMarkovianSolution,
    p0: float,
    tau: float,
    n_steps: int,
    n_paths: int,
    seed: int,
    x0: float = 0.0,
    antithetic: bool = False,
) -> tuple[Array, Array]:
    """SYNTHETIC risk-neutral (X, p) pathwise log returns (Prop. 5.3).

    Returns (log_returns, discount_factors).  ``antithetic=True`` pairs
    each Gaussian draw with its negation (the Sec. 5.5 variance reduction);
    with it enabled ``n_paths`` is rounded up to an even count.
    """
    sol._require_eta_positive()
    _positive_scalar(tau, "tau")
    pp0 = _finite_scalar(p0, "p0")
    if not 0.0 < pp0 < 1.0:
        raise ValueError("p0 must lie in (0, 1)")
    if int(n_steps) < 1 or int(n_paths) < 8:
        raise ValueError("n_steps >= 1 and n_paths >= 8 required")
    xx0 = _finite_scalar(x0, "x0")
    rng = np.random.default_rng(int(seed))
    n_m = int(n_paths)
    if antithetic:
        half = rng.standard_normal((int(n_steps), (n_m + 1) // 2))
        z = np.concatenate([half, -half], axis=1)
    else:
        z = rng.standard_normal((int(n_steps), n_m))
    return _mc_paths_q(sol, pp0, tau, z, xx0)


def sample_skewness(x: object) -> tuple[float, float]:
    """Moment skewness m3 / m2^{3/2} with normal-approx se = sqrt(6/n).

    A standardized third moment — a proper-shape diagnostic, not a score.
    Fail-closed on degenerate input.
    """
    v = np.asarray(x, dtype=float).reshape(-1)
    if v.size < 8 or not bool(np.all(np.isfinite(v))):
        raise ValueError("x must be finite with >= 8 observations")
    m = v - v.mean()
    m2 = float(np.mean(m * m))
    if m2 <= 0.0:
        raise ValueError("degenerate (zero-variance) sample")
    m3 = float(np.mean(m**3))
    return float(m3 / m2**1.5), float(math.sqrt(6.0 / v.size))


def mc_log_return_skewness(
    sol: BeliefMarkovianSolution,
    p0: float,
    tau: float,
    n_steps: int,
    n_paths: int,
    seed: int,
) -> tuple[float, float]:
    """SYNTHETIC MC conditional risk-neutral log-return skewness (Prop. 5.5 check)."""
    ret, _ = mc_log_returns_q(sol, p0, tau, n_steps, n_paths, seed)
    return sample_skewness(ret)


def mc_option_call(
    sol: BeliefMarkovianSolution,
    p0: float,
    strike: float,
    tau: float,
    s0: float = 1.0,
    n_steps: int = 128,
    n_paths: int = 65_536,
    seed: int = 7,
) -> tuple[float, float]:
    """SYNTHETIC MC price of the European call (5.9) under Q.

    E^Q[exp(-int r_f ds) (S_tau - K)^+] with pathwise discounting and the
    Sec. 5.5 antithetic variates; returns (price, MC standard error).
    """
    k = _positive_scalar(strike, "strike")
    s = _positive_scalar(s0, "s0")
    ret, disc = mc_log_returns_q(
        sol, p0, tau, n_steps, n_paths, seed, x0=math.log(s), antithetic=True
    )
    pay = disc * np.maximum(s * np.exp(ret) - k, 0.0)
    price = float(np.mean(pay))
    se = float(np.std(pay, ddof=1) / math.sqrt(pay.size))
    return price, se


# ---------------------------------------------------------------------------
# European option-pricing PDE (Thm. 5.4) — Crank-Nicolson, 9-point stencil
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OptionSurface:
    """Crank-Nicolson solution of (5.10) on the (X, p) grid."""

    x_grid: Array
    p_grid: Array
    values: Array  # shape (n_x, n_p): C(X_i, p_j, tau)
    tau: float
    strike: float

    def price(self, s0: float, p0: float) -> float:
        """Bilinear interpolation at (log s0, p0)."""
        x = math.log(_positive_scalar(s0, "s0"))
        pp = _finite_scalar(p0, "p0")
        if not 0.0 < pp < 1.0:
            raise ValueError("p0 must lie in (0, 1)")
        if not self.x_grid[0] <= x <= self.x_grid[-1]:
            raise ValueError("s0 outside the solved X grid")
        i1 = int(np.searchsorted(self.x_grid, x, side="right")) - 1
        i1 = min(max(i1, 0), self.x_grid.size - 2)
        j1 = int(np.searchsorted(self.p_grid, pp, side="right")) - 1
        j1 = min(max(j1, 0), self.p_grid.size - 2)
        wx = (x - self.x_grid[i1]) / (self.x_grid[i1 + 1] - self.x_grid[i1])
        wp = (pp - self.p_grid[j1]) / (self.p_grid[j1 + 1] - self.p_grid[j1])
        c = self.values
        return float(
            (1 - wx) * (1 - wp) * c[i1, j1]
            + wx * (1 - wp) * c[i1 + 1, j1]
            + (1 - wx) * wp * c[i1, j1 + 1]
            + wx * wp * c[i1 + 1, j1 + 1]
        )


def _option_operator(
    sol: BeliefMarkovianSolution,
    pg: Array,
    hx: float,
    hp: float,
    n_x: int,
) -> csr_matrix:
    """Assemble the discrete generator L^Q - r_f of (5.10).

    Row m = i*n_p + j.  Interior-p rows use the 9-point stencil; the two
    p-boundary rows implement the Thm. 5.4 reduced equations with
    one-sided dp (lam^{21} dp at p = 0, -lam^{12} dp at p = 1).  The X-edge
    rows are left identically zero — their values are Dirichlet data
    substituted into the CN system separately.
    """
    n_p = pg.size
    e = sol.econ
    sig_s = sol.stock_vol(pg)
    chi = belief_vol(pg, e)
    btil = sol.rn_belief_drift(pg)
    rf = sol.marginal_short_rate(pg)
    bx = sol.log_stock_drift_q(pg)

    m_size = n_x * n_p
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []

    def add(m: int, k: int, v: float) -> None:
        rows.append(m)
        cols.append(k)
        vals.append(v)

    for i in range(1, n_x - 1):
        base = i * n_p
        for j in range(n_p):
            m = base + j
            s2 = 0.5 * sig_s[j] ** 2
            cx = bx[j]
            # -r_f C + 1/2 sigma_S^2 dXX + b_X dX (shared by interior and
            # p-boundary rows; the boundary rows differ only in the p-part)
            add(m, m, -rf[j] - 2.0 * s2 / hx**2)
            add(m, m + n_p, s2 / hx**2 + cx / (2.0 * hx))
            add(m, m - n_p, s2 / hx**2 - cx / (2.0 * hx))
            if 0 < j < n_p - 1:
                sc = sig_s[j] * chi[j]
                cp = 0.5 * chi[j] ** 2
                bp = btil[j]
                # 1/2 chi^2 dpp + Btilde dp (centered)
                add(m, m + 1, cp / hp**2 + bp / (2.0 * hp))
                add(m, m - 1, cp / hp**2 - bp / (2.0 * hp))
                add(m, m, -2.0 * cp / hp**2)
                # sigma_S chi dXp: 7-point stencil for the cross derivative
                # (sc >= 0 always here since sigma_S > 0 and chi >= 0):
                # dXp ~ (C_{i+1,j+1} - C_{i+1,j} - C_{i,j+1} + 2 C_{i,j}
                #        - C_{i,j-1} - C_{i-1,j} + C_{i-1,j-1}) / (2 hx hp)
                # — the monotone-oriented discretization; it leaves only the
                # two +sc/(2 hx hp) corners with positive weight and visibly
                # damps the CN kink oscillation vs the centered 4-point form.
                w = sc / (2.0 * hx * hp)
                add(m, m + n_p + 1, +w)
                add(m, m + n_p, -w)
                add(m, m + 1, -w)
                add(m, m, +2.0 * w)
                add(m, m - 1, -w)
                add(m, m - n_p, -w)
                add(m, m - n_p - 1, +w)
            elif j == 0:
                # reduced boundary equation at p = 0: + lam^{21} dp (one-sided)
                add(m, m + 1, e.lam_up / hp)
                add(m, m, -e.lam_up / hp)
            else:
                # reduced boundary equation at p = 1: - lam^{12} dp (one-sided)
                add(m, m, -e.lam_down / hp)
                add(m, m - 1, e.lam_down / hp)
    return csr_matrix(
        (np.asarray(vals), (np.asarray(rows), np.asarray(cols))),
        shape=(m_size, m_size),
    )


def option_call_cn(
    sol: BeliefMarkovianSolution,
    strike: float,
    tau: float,
    s0: float = 1.0,
    n_x: int = 161,
    n_p: int = 81,
    n_t: int = 60,
    x_half_width: float | None = None,
) -> OptionSurface:
    """European call via Crank-Nicolson on (5.10) with Rannacher smoothing.

    X domain is [log s0 - hw, log s0 + hw] with hw defaulting to
    6 sigma_S,max sqrt(tau) + max|b_X| tau + 0.5 (expanded so both s0 and K
    lie inside).  p-boundary rows use the reduced Thm. 5.4 equations.  The
    X edges use time-invariant Dirichlet values localized in p: C = 0 at
    X_lo and C = max(e^X e^{-q_S(p) tau} - K e^{-r_f(p) tau}, 0) at X_hi —
    the frozen-coefficient asymptotics; an explicit approximation, kept far
    from the strike by the wide domain.  The first two steps are backward
    Euler (Rannacher) to damp the payoff kink.  The cross derivative uses
    the 7-point stencil (sigma_S chi >= 0), which is the monotone-oriented
    discretization; residual deep-OTM oscillation is bounded by ~1e-4 and
    shrinks under grid refinement (measured: ~3e-5 at 121x61x50, ~1e-5 at
    201x101x100 on the Sec. 5.5 economy).  Deterministic; fail-closed on
    eta <= 0 and degenerate grids/parameters.
    """
    sol._require_eta_positive()
    k = _positive_scalar(strike, "strike")
    t = _positive_scalar(tau, "tau")
    s = _positive_scalar(s0, "s0")
    nxi, npi, nti = int(n_x), int(n_p), int(n_t)
    if nxi < 9 or npi < 9 or nti < 1:
        raise ValueError("n_x >= 9, n_p >= 9 and n_t >= 1 required")

    pg = np.linspace(0.0, 1.0, npi)
    sig_max = float(sol.stock_vol(sol.p).max())
    bx_abs = float(np.abs(sol.log_stock_drift_q(pg)).max())
    hw = (
        6.0 * sig_max * math.sqrt(t) + bx_abs * t + 0.5
        if x_half_width is None
        else _positive_scalar(x_half_width, "x_half_width")
    )
    x0 = math.log(s)
    hw = max(hw, abs(x0 - math.log(k)) + 0.25)
    x = np.linspace(x0 - hw, x0 + hw, nxi)
    hx = float(x[1] - x[0])
    hp = 1.0 / (npi - 1)
    dtau = t / nti

    lmat = _option_operator(sol, pg, hx, hp, nxi)
    ident = identity(nxi * npi, format="csr")
    a_left = ident - 0.5 * dtau * lmat
    a_right = ident + 0.5 * dtau * lmat
    # Dirichlet X edges: overwrite the boundary rows of A_left with the
    # identity row so C_edge^{n+1} = rhs_edge (a fixed value); A_right's
    # contribution at those rows is replaced by the explicit assignment.
    rf_v = sol.marginal_short_rate(pg)
    qv = sol.dividend_yield(pg)
    bc_hi = np.maximum(math.exp(x[-1]) * np.exp(-qv * t) - k * np.exp(-rf_v * t), 0.0)

    lil = a_left.tolil()
    edge_rows = list(range(npi)) + list(range((nxi - 1) * npi, nxi * npi))
    for row in edge_rows:
        lil.rows[row] = [row]
        lil.data[row] = [1.0]
    a_left = lil.tocsr()
    lu_left = factorized(a_left.tocsc())

    cval = np.maximum(np.exp(x) - k, 0.0)[:, None] * np.ones((1, npi))
    cval = cval.reshape(-1)
    cval[:npi] = 0.0
    cval[(nxi - 1) * npi :] = bc_hi

    for step in range(nti):
        rhs_vec = a_right @ cval
        rhs_vec[:npi] = 0.0
        rhs_vec[(nxi - 1) * npi :] = bc_hi
        if step < 2:  # Rannacher damping at the payoff kink
            rhs_vec = 0.5 * (rhs_vec + cval)
        cval = np.asarray(lu_left(rhs_vec), dtype=float)

    surface = cval.reshape(nxi, npi)
    return OptionSurface(x_grid=x, p_grid=pg, values=surface, tau=t, strike=k)


# ---------------------------------------------------------------------------
# Bench helper — flat dict[str, float], seeded, ~seconds
# ---------------------------------------------------------------------------


def bench_hidden_markov_equilibrium(seed: int = 2026) -> dict[str, float]:
    """Seeded SYNTHETIC bench on the paper's Sec. 5.5 calibration.

    Solves the pricing BVP, runs the diagnostics, filters a simulated
    dividend path, checks the leading skewness against a seeded MC, and
    prices one ATM call by both Crank-Nicolson and Monte-Carlo.  Returns a
    flat ``dict[str, float]`` of diagnostics only — solver-tolerance and
    proper-score material, no headline performance metric.
    """
    econ = calibrate_option_economy()
    sol = solve_price_dividend_ratio(econ, n_grid=801)
    diag = pricing_diagnostics(sol)

    sim = simulate_dividends(econ, n_steps=2_000, dt=0.02, seed=seed)
    path = wonham_filter(sim.levels, sim.dt, econ)
    _sd_f, _sd_s = float(np.std(path.filtered)), float(np.std(sim.states[1:]))
    corr = (
        float(np.corrcoef(path.filtered, sim.states[1:])[0, 1])
        if _sd_f > 1e-12 and _sd_s > 1e-12
        else 0.0
    )

    lead = float(sol.leading_skewness(0.30, 0.02))
    mc_sk, mc_se = mc_log_return_skewness(
        sol, 0.30, tau=0.02, n_steps=64, n_paths=30_000, seed=seed
    )
    skew_err = abs(lead - mc_sk) / max(abs(lead), 1e-9)

    surf = option_call_cn(sol, strike=1.0, tau=0.10, s0=1.0, n_x=121, n_p=61, n_t=50)
    cn = surf.price(1.0, 0.5)
    mc_p, mc_se_p = mc_option_call(
        sol, 0.5, 1.0, 0.10, s0=1.0, n_steps=64, n_paths=30_000, seed=seed
    )
    opt_err = abs(cn - mc_p) / max(mc_p, 1e-9)

    rf_grid = sol.marginal_short_rate(np.linspace(0.0, 1.0, 201))
    return {
        "synthetic_phi_p0": float(sol.phi[0]),
        "synthetic_phi_p_half": float(sol.price_dividend_ratio(0.5)),
        "synthetic_phi_p1": float(sol.phi[-1]),
        "synthetic_phi_min": diag["phi_min"],
        "synthetic_phi_max": diag["phi_max"],
        "synthetic_ode_residual_max": diag["ode_residual_max"],
        "synthetic_bc_residual_max": diag["bc_residual_max"],
        "synthetic_factor_flatness": diag["factor_flatness"],
        "synthetic_sigmaS_max": diag["sigmaS_max"],
        "synthetic_p_star": diag["p_star"],
        "synthetic_sigmaS_endpoint": diag["sigmaS_endpoint0"],
        "synthetic_rf_min": float(rf_grid.min()),
        "synthetic_rf_max": float(rf_grid.max()),
        "synthetic_skew_lead_p030": lead,
        "synthetic_skew_mc_p030": float(mc_sk),
        "synthetic_skew_mc_se_p030": float(mc_se),
        "synthetic_skew_rel_err": float(skew_err),
        "synthetic_option_cn_atm": float(cn),
        "synthetic_option_mc_atm": float(mc_p),
        "synthetic_option_mc_se": float(mc_se_p),
        "synthetic_option_rel_err": float(opt_err),
        "synthetic_filter_state_corr": corr,
        "synthetic_innovation_var": float(np.var(path.innovations)),
        "synthetic_pit_mean": float(np.mean(path.pit)),
        "synthetic_filter_loglik": float(path.log_likelihood),
        "synthetic_n_iter": float(sol.n_iter),
    }
