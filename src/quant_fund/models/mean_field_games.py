"""Mean Field Games for market microstructure and crowded trading.

Implements the Cardaliaguet & Lehalle (2018) trade-crowding MFG via the
linear-quadratic (LQ) spectral decomposition.  The equilibrium is
characterised by a forward-backward ODE system:

    q*'(t) = α*(t),  q*(0) = Q₀
    α*(t,q) = −∂_q v / (2κ) = −(h₂(t) q + ½ h₁(t)) / κ

where h₂(t) solves a decoupled Riccati equation and (q*, h₁) form a
coupled FBSDE solved via the Riccati method (co-state/state ratio P(t)).
No fixed-point iteration is needed for the LQ structure.

References
----------
- Cardaliaguet & Lehalle (2018). "Mean field game of controls and an
  application to trade crowding." *Mathematical Finance* 28, 265–293.
- Carmona, Fouque & Sun (2015). "Mean field games and systemic risk."
  *Communications in Mathematical Sciences* 13, 911–933.
- Guéant, Lasry & Lions (2011). "Mean field games and applications."
  *Paris-Princeton Lectures on Mathematical Finance*, Springer.
- Huang, Malhamé & Caines (2006). "Large population stochastic dynamic
  games: closed-loop McKean-Vlasov systems and the Nash certainty
  equivalence principle." *IEEE TAC* 51, 1188–1213.

The implementation is a SYNTHETIC research tool: no live-trading claims,
no broker connectivity; output is a mathematical object only.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class MFGConfig(NamedTuple):
    """Parameter bundle for the trade-crowding MFG.

    Attributes
    ----------
    Q0 : initial inventory (positive = long position to liquidate).
    T : time horizon.
    kappa : temporary-impact penalty per unit trading speed squared.
    gamma : permanent-impact coupling strength (mean-field).
    psi : running inventory penalty (risk-aversion × variance / 2).
    terminal_penalty : penalty A on terminal inventory (A → ∞ = hard
                       liquidation constraint).
    n_t : number of time steps.
    n_q : number of inventory grid points (for output arrays).
    q_min, q_max : inventory grid bounds (auto-detected if None).
    """

    Q0: float
    T: float
    kappa: float
    gamma: float
    psi: float
    terminal_penalty: float
    n_t: int = 100
    n_q: int = 80
    q_min: float | None = None
    q_max: float | None = None


class MFGSolution(NamedTuple):
    """Equilibrium solution of the trade-crowding MFG.

    All attributes are SYNTHETIC mathematical objects — no market data,
    no live-trading claims.

    Attributes
    ----------
    t_grid : time grid, shape (n_t+1,).
    q_grid : inventory grid, shape (n_q,).
    value : HJB value v(t,q) = h₂ q² + h₁ q + h₀ on (n_t+1, n_q).
    density : FP density m(t,q) on (n_t+1, n_q).
    alpha : optimal trading rate α*(t,q) on (n_t+1, n_q).
    mu : aggregate rate μ̄_t, shape (n_t+1,).
    optimal_trajectory : q*(t), shape (n_t+1,).
    n_fp_iters : number of fixed-point iterations (1 for LQ solver).
    fp_residual : final residual (0 for LQ spectral solver).
    cost : equilibrium expected cost.
    solitary_cost : expected cost when γ = 0.
    crowding_cost_ratio : cost / solitary_cost.
    h2 : Riccati coefficient h₂(t), shape (n_t+1,).
    h1 : drift coefficient h₁(t), shape (n_t+1,).
    h0 : intercept h₀(t), shape (n_t+1,).
    """

    t_grid: Array
    q_grid: Array
    value: Array
    density: Array
    alpha: Array
    mu: Array
    optimal_trajectory: Array
    n_fp_iters: int
    fp_residual: float
    cost: float
    solitary_cost: float
    crowding_cost_ratio: float
    h2: Array
    h1: Array
    h0: Array


# ---------------------------------------------------------------------------
# h₂ Riccati solver (analytic)
# ---------------------------------------------------------------------------

_MAX_EXP = 700.0  # safe upper bound for math.exp


def _solve_h2(t_grid: Array, kappa: float, psi: float, A: float) -> Array:
    """Solve h₂' = h₂²/κ − ψ, h₂(T) = A, backward in time.

    Analytic solution using separation of variables.  Returns h₂ on t_grid.
    """
    n_t = len(t_grid) - 1
    T = float(t_grid[-1])
    h2 = np.empty(n_t + 1, dtype=np.float64)
    h2[-1] = A

    omega = math.sqrt(max(psi / kappa, 0.0))
    k_omega = kappa * omega

    for k in range(n_t - 1, -1, -1):
        tau = T - float(t_grid[k])

        if omega == 0.0:
            # ψ = 0 → h₂' = h₂²/κ → h₂ = κ / (κ/A + τ)
            if A == 0.0:
                h2[k] = 0.0
            else:
                h2[k] = kappa / (kappa / A + tau)
            continue

        if abs(A - k_omega) < 1e-15:
            h2[k] = k_omega
            continue

        two_w_tau = 2.0 * omega * tau
        if two_w_tau > _MAX_EXP:
            # Large τ → h₂ ≈ κω (the stable equilibrium)
            h2[k] = k_omega
            continue

        # (h - κω) / (h + κω) = (A - κω) / (A + κω) · e^{−2ωτ}
        ratio = (A - k_omega) / (A + k_omega) * math.exp(-two_w_tau)

        if abs(ratio - 1.0) < 1e-15:
            # Near blow-up → large h₂
            h2[k] = 1e15
        else:
            h2[k] = k_omega * (1.0 + ratio) / (1.0 - ratio)

    return h2


# ---------------------------------------------------------------------------
# FBSDE solver via Riccati method
# ---------------------------------------------------------------------------


def _solve_fbsde(
    t_grid: Array,
    Q0: float,
    kappa: float,
    gamma: float,
    h2: Array,
) -> tuple[Array, Array, Array]:
    """Solve the coupled (q*, h₁, h₀) FBSDE.

    Forward:  q*' = −(h₂/κ) q* − h₁/(2κ),           q*(0) = Q₀
    Backward: h₁' = (h₂/κ) h₁ + γ·μ̄,                 h₁(T) = 0
              h₀' = h₁²/(4κ),                         h₀(T) = 0
              μ̄(t) = −(h₂/κ) q*(t) − h₁(t)/(2κ)

    Solved via the Riccati ansatz h₁(t) = P(t)·q*(t) where P satisfies
    a backward Riccati ODE.  q* is then forward-integrated.
    """
    n_t = len(t_grid) - 1
    dt = t_grid[1] - t_grid[0]

    if gamma == 0.0:
        q_star = np.empty(n_t + 1, dtype=np.float64)
        q_star[0] = Q0
        for k in range(n_t):
            rate = -h2[k] / kappa
            q_star[k + 1] = q_star[k] * math.exp(rate * dt)
        return q_star, np.zeros(n_t + 1), np.zeros(n_t + 1)

    # --- Riccati for P(t), backward ----------------------------------------
    # P' = −γ h₂/κ + (2h₂/κ − γ/(2κ))·P + P²/(2κ),   P(T) = 0
    P = np.empty(n_t + 1, dtype=np.float64)
    P[-1] = 0.0

    for k in range(n_t - 1, -1, -1):
        h = h2[k]
        # P' = c0 + c1·P + c2·P²
        c0 = -gamma * h / kappa
        c1 = 2.0 * h / kappa - 0.5 * gamma / kappa
        c2 = 0.5 / kappa

        # Backward Euler stepping back in time: P_k = P_{k+1} − dt·f(P_k)
        # → c2·P_k² + (c1 + 1/dt)·P_k + (c0 − P_{k+1}/dt) = 0
        a = c2
        b = c1 + 1.0 / dt
        c_val = c0 - P[k + 1] / dt

        disc = b * b - 4.0 * a * c_val
        if disc < 0:
            raise ValueError(
                f"Riccati discriminant negative at t={t_grid[k]:.4f}. "
                f"Parameters may be beyond the continuum region "
                f"for the LQ MFG (try smaller gamma or larger kappa)."
            )
        sqrt_disc = math.sqrt(disc)

        root1 = (-b + sqrt_disc) / (2.0 * a)
        root2 = (-b - sqrt_disc) / (2.0 * a)
        # Select root continuous from P_{k+1} (closer to it)
        P[k] = root1 if abs(root1 - P[k + 1]) < abs(root2 - P[k + 1]) else root2

        if not np.isfinite(P[k]):
            raise ValueError(
                f"Riccati for P(t) diverged at t={t_grid[k]:.4f}. "
                f"Parameters may be in a no-equilibrium region."
            )

    # --- Forward integration of q* -----------------------------------------
    q_star = np.empty(n_t + 1, dtype=np.float64)
    q_star[0] = Q0
    for k in range(n_t):
        rate = -(h2[k] + 0.5 * P[k]) / kappa
        q_star[k + 1] = q_star[k] * math.exp(rate * dt)

    # --- h₁ = P·q* --------------------------------------------------------
    h1 = P * q_star

    # --- Forward quadrature for h₀ -----------------------------------------
    # h₀' = h₁²/(4κ), h₀(T) = 0 → h₀(t) = −∫_t^T h₁²/(4κ) ds
    h0 = np.empty(n_t + 1, dtype=np.float64)
    h0[-1] = 0.0
    for k in range(n_t - 1, -1, -1):
        h0[k] = h0[k + 1] - dt * h1[k] ** 2 / (4.0 * kappa)

    return q_star, h1, h0


# ---------------------------------------------------------------------------
# Grid helpers
# ---------------------------------------------------------------------------


def _build_q_grid(cfg: MFGConfig) -> tuple[Array, float]:
    """Build inventory grid, returning (grid, dq)."""
    if cfg.q_min is not None and cfg.q_max is not None:
        q_grid = np.linspace(cfg.q_min, cfg.q_max, cfg.n_q, dtype=np.float64)
    else:
        Q0 = cfg.Q0
        margin = max(abs(Q0) * 1.5, 0.5)
        q_min = min(-margin, Q0 - margin)
        q_max = max(margin, Q0 + margin)
        q_grid = np.linspace(q_min, q_max, cfg.n_q, dtype=np.float64)
    return q_grid, q_grid[1] - q_grid[0]


def _dirac_on_grid(q_grid: Array, q_val: float, dq: float) -> Array:
    """Place unit-mass density at nearest grid point to q_val."""
    idx = int(np.argmin(np.abs(q_grid - q_val)))
    idx = max(0, min(idx, len(q_grid) - 1))
    m = np.zeros_like(q_grid)
    m[idx] = 1.0 / dq
    return m


# ---------------------------------------------------------------------------
# Cost computation
# ---------------------------------------------------------------------------


def _cost_trajectory(
    t_grid: Array,
    q_star: Array,
    h2: Array,
    h1: Array,
    kappa: float,
    gamma: float,
    psi: float,
    A: float,
) -> float:
    """Expected cost along q*(t) under coefficients (h₂, h₁)."""
    n_t = len(t_grid) - 1
    dt = t_grid[1] - t_grid[0]

    # α(t) = q*'(t) via backward difference (stable)
    alpha_t = np.empty(n_t + 1, dtype=np.float64)
    alpha_t[0] = (q_star[1] - q_star[0]) / dt
    for k in range(1, n_t):
        alpha_t[k] = (q_star[k + 1] - q_star[k - 1]) / (2.0 * dt)
    alpha_t[-1] = (q_star[-1] - q_star[-2]) / dt

    # μ̄(t)
    mu_t = -(h2 * q_star + 0.5 * h1) / kappa

    integrand = kappa * alpha_t**2 + psi * q_star**2 - gamma * q_star * mu_t
    cost = 0.5 * dt * np.sum(integrand[:-1] + integrand[1:])
    cost += A * q_star[-1] ** 2
    return float(cost)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _validate_config(cfg: MFGConfig) -> None:
    """Validate MFGConfig parameters.  Raises ValueError on bad input."""
    if not np.isfinite(cfg.Q0):
        raise ValueError("Q0 must be finite")
    if cfg.T <= 0 or not np.isfinite(cfg.T):
        raise ValueError("T must be finite and positive")
    if cfg.kappa <= 0 or not np.isfinite(cfg.kappa):
        raise ValueError("kappa must be finite and positive")
    if not np.isfinite(cfg.gamma) or cfg.gamma < 0:
        raise ValueError("gamma must be finite and non-negative")
    if not np.isfinite(cfg.psi) or cfg.psi < 0:
        raise ValueError("psi must be finite and non-negative")
    if not np.isfinite(cfg.terminal_penalty) or cfg.terminal_penalty < 0:
        raise ValueError("terminal_penalty must be finite and non-negative")
    if int(cfg.n_t) < 3:
        raise ValueError("n_t must be integer >= 3")
    if int(cfg.n_q) < 5:
        raise ValueError("n_q must be integer >= 5")
    q_min = cfg.q_min
    q_max = cfg.q_max
    if q_min is not None and q_max is not None and q_min >= q_max:
        raise ValueError(f"q_min ({q_min}) must be < q_max ({q_max})")


def solve_mfg_trade_crowding(cfg: MFGConfig) -> MFGSolution:
    """Solve the Cardaliaguet–Lehalle (2018) trade-crowding MFG.

    Uses the LQ spectral decomposition.  The HJB value function is
    v(t,q) = h₂ q² + h₁ q + h₀ and the Fokker–Planck density is a Dirac
    at the optimal trajectory q*(t).  The FBSDE reduces to a Riccati ODE
    for the co-state/state ratio P(t) — no fixed-point iteration needed.

    Parameters
    ----------
    cfg : MFGConfig

    Returns
    -------
    MFGSolution (SYNTHETIC only — mathematical object, not market
    evidence or live-trading advice).
    """
    _validate_config(cfg)

    Q0 = float(cfg.Q0)
    T = float(cfg.T)
    kappa = float(cfg.kappa)
    gamma = float(cfg.gamma)
    psi = float(cfg.psi)
    A = float(cfg.terminal_penalty)
    n_t = int(cfg.n_t)
    n_q = int(cfg.n_q)

    t_grid = np.linspace(0.0, T, n_t + 1, dtype=np.float64)
    q_grid, dq = _build_q_grid(cfg)

    # --- Solitary (γ = 0) --------------------------------------------------
    h2_sol = _solve_h2(t_grid, kappa, psi, A)
    q_sol, h1_sol, h0_sol = _solve_fbsde(t_grid, Q0, kappa, 0.0, h2_sol)

    # --- Equilibrium (γ ≥ 0) -----------------------------------------------
    h2_eq = _solve_h2(t_grid, kappa, psi, A)
    q_star, h1_eq, h0_eq = _solve_fbsde(t_grid, Q0, kappa, gamma, h2_eq)

    # --- Build grid output arrays ------------------------------------------
    value = np.empty((n_t + 1, n_q), dtype=np.float64)
    alpha_arr = np.empty((n_t + 1, n_q), dtype=np.float64)
    density = np.empty((n_t + 1, n_q), dtype=np.float64)
    mu = np.empty(n_t + 1, dtype=np.float64)

    for k in range(n_t + 1):
        value[k] = h2_eq[k] * q_grid**2 + h1_eq[k] * q_grid + h0_eq[k]
        alpha_arr[k] = -(2.0 * h2_eq[k] * q_grid + h1_eq[k]) / (2.0 * kappa)
        density[k] = _dirac_on_grid(q_grid, q_star[k], dq)
        mu[k] = -(h2_eq[k] * q_star[k] + 0.5 * h1_eq[k]) / kappa

    # --- Costs -------------------------------------------------------------
    cost = _cost_trajectory(t_grid, q_star, h2_eq, h1_eq, kappa, gamma, psi, A)
    solitary_cost = _cost_trajectory(t_grid, q_sol, h2_sol, h1_sol, kappa, 0.0, psi, A)
    crowding_ratio = cost / solitary_cost if solitary_cost > 1e-20 else 1.0

    return MFGSolution(
        t_grid=t_grid,
        q_grid=q_grid,
        value=value,
        density=density,
        alpha=alpha_arr,
        mu=mu,
        optimal_trajectory=q_star,
        n_fp_iters=1,
        fp_residual=0.0,
        cost=cost,
        solitary_cost=solitary_cost,
        crowding_cost_ratio=crowding_ratio,
        h2=h2_eq,
        h1=h1_eq,
        h0=h0_eq,
    )


# ---------------------------------------------------------------------------
# Almgren–Chriss benchmark for ε→0 verification
# ---------------------------------------------------------------------------


def ac_benchmark_trajectory(
    Q0: float,
    T: float,
    n_slices: int,
    kappa: float,
    psi: float,
) -> Array:
    """Closed-form optimal trading rate in the γ = 0, A → ∞ limit.

    This is the Almgren–Chriss schedule:
        q(t) = Q₀ · sinh(ω(T−t)) / sinh(ω T),   ω = √(ψ / κ).
    Returns α̂(t) = dq/dt at each slice midpoint (negative: liquidation).
    """
    if psi <= 0:
        raise ValueError("psi must be positive for the AC benchmark")
    if not np.isfinite(Q0):
        raise ValueError("Q0 must be finite")
    if int(n_slices) < 1:
        raise ValueError("n_slices must be >= 1")

    omega = math.sqrt(psi / kappa)
    omega_T = omega * T
    t = np.linspace(0, T, n_slices + 1, dtype=np.float64)
    dt = t[1] - t[0]

    if omega_T > _MAX_EXP:
        alpha = np.zeros(n_slices, dtype=np.float64)
        alpha[0] = -Q0 / dt
        return alpha

    sinh_omega_T = math.sinh(omega_T)
    alpha = np.empty(n_slices, dtype=np.float64)
    for i in range(n_slices):
        t_mid = 0.5 * (t[i] + t[i + 1])
        alpha[i] = float(-Q0 * omega * math.cosh(omega * (T - t_mid)) / sinh_omega_T)

    return alpha


def mfg_to_ac_alpha(sol: MFGSolution, n_slices: int) -> Array:
    """Extract the MFG optimal trading rate at q*(t) for comparison.

    Interpolates α*(t, q*(t)) onto ``n_slices`` equispaced points
    at the midpoints of the slice intervals.
    """
    t_grid = sol.t_grid
    T = float(t_grid[-1])
    t_eval = np.linspace(0, T, n_slices + 1, dtype=np.float64)
    alpha_slices = np.empty(n_slices, dtype=np.float64)

    for i in range(n_slices):
        t_mid = 0.5 * (t_eval[i] + t_eval[i + 1])
        k = min(int(t_mid / (t_grid[1] - t_grid[0])), len(t_grid) - 1)
        q_val = sol.optimal_trajectory[k]
        alpha_slices[i] = float(np.interp(q_val, sol.q_grid, sol.alpha[k]))

    return alpha_slices
