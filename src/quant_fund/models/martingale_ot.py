"""Martingale Optimal Transport (MOT) — model-free option pricing bounds.

Given market prices of vanilla options (which fix the risk-neutral marginal
distribution of the underlying at maturity via Breeden–Litzenberger), the
tightest arbitrage-free bounds for an exotic option are obtained by solving
an optimal transport problem with a martingale constraint. The dual
formulation is a semi-static superhedging problem: the optimal dual
variables are the cheapest portfolio of static vanilla positions plus
dynamic delta hedging that dominates the exotic payoff.

One-period discrete-time formulation (Beiglböck, Henry-Labordère & Penkner
2013, Finance & Stochastics 17, arXiv:1106.5929; Henry-Labordère 2017,
*Model-Free Hedging: A Martingale Optimal Transport Viewpoint*, CRC Press):

Primal (upper bound — maximise exotic expected payoff):

    max_{π >= 0}  Σ_j f_j π_j
    s.t.  Σ_j π_j = 1,
          Σ_j x_j π_j = F,
          Σ_j (x_j - K_i)^+ π_j = C_i,  i = 1...M,

where ``x_j`` are discrete terminal-spot states, ``f_j = f(x_j)`` is the
exotic payoff, ``F`` is the forward, ``K_i`` are market strikes, and
``C_i`` are undiscounted expected call payoffs (market price × e^{rT}).

Dual (upper bound — cheapest superhedge):

    min_{ν, η, α}  ν + η F + Σ_i α_i C_i
    s.t.  ν + η x_j + Σ_i α_i (x_j - K_i)^+ >= f_j,  j = 1...N.

Here ``ν`` is the cash (numeraire) position, ``η`` is the forward/delta
position, and ``α_i`` are static positions in vanilla call ``i``. The
superhedging portfolio value dominates the exotic payoff in every state.

For the lower bound (minimise exotic expected payoff), both primal and dual
inequalities are reversed (dual becomes a subhedge with ``<=``).

Continuous-time MOT (Dolinsky & Soner 2014, Probab. Theory Relat. Fields
160, arXiv:1208.4922) extends the duality to path-dependent European
options via Progess theory. The discrete LP here is the canonical
computational building block; the continuous-time theory guarantees that
the discrete bounds converge to the true model-free bounds as the strike
and state grids refine.

Honesty contract: outputs are proper pricing bounds and superhedging costs.
No Sharpe/Sortino/P&L content; no live-trading claims. SYNTHETIC test data
is labelled as such.

References
----------
- Beiglböck, M., Henry-Labordère, P. & Penkner, F. (2013). Model-independent
  bounds for option prices — a mass transport approach. *Finance and
  Stochastics* 17, 439-476. arXiv:1106.5929 — Theorem 1 (no-duality-gap).
- Dolinsky, Y. & Soner, H. M. (2014). Martingale optimal transport and
  robust hedging in continuous time. *Probability Theory and Related Fields*
  160(1-2), 391–427. arXiv:1208.4922 — Section 2 (quasi-sure duality).
- Henry-Labordère, P. (2017). *Model-Free Hedging: A Martingale Optimal
  Transport Viewpoint*. CRC Press. — Chapter 2 (one-period MOT LP).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linprog
from scipy.stats import norm

Array = NDArray[np.float64]

__all__ = [
    "MOTBounds",
    "MOTHedgePortfolio",
    "generate_synthetic_vanillas",
    "solve_mot_bounds",
    "solve_mot_lower_bound",
    "solve_mot_upper_bound",
    "variance_swap_mot_bounds",
]

_DEFAULT_RNG = np.random.default_rng(42)
_DUAL_TOL = 1e-6  # acceptable primal-dual gap (fraction of objective scale)
_EPS = np.finfo(np.float64).eps


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MOTHedgePortfolio:
    """Dual MOT variables — the semi-static superhedging (or subhedging) portfolio.

    ``cash`` is the numeraire position (``ν``). ``forward_position`` is the
    delta position in the underlying forward contract (``η``).
    ``option_positions`` are the static positions in each vanilla call
    strike, in the same order as the input ``strikes`` array. Together,

        hedge_value(x) = cash + forward_position * x
                         + Σ_i option_positions[i] * max(x - K_i, 0).

    For an upper-bound solve, ``hedge_cost`` = ``cash + forward_position*F
    + Σ_i option_positions[i]*C_i`` is the superhedging cost; the portfolio
    satisfies ``hedge_value(x_j) >= exotic_payoff[j]`` at every grid point.
    """

    cash: float
    forward_position: float
    option_positions: Array
    hedge_cost: float


@dataclass(frozen=True)
class MOTBounds:
    """Bounds on an exotic option's expected payoff from MOT.

    All values are *undiscounted* expected payoffs (multiply by ``exp(-rT)``
    to get present-value prices). ``upper_bound`` and ``lower_bound`` are
    the optimal primal values; ``upper_hedge`` and ``lower_hedge`` are the
    corresponding dual portfolios.

    ``upper_dual_gap`` = ``upper_bound - upper_hedge.hedge_cost`` and
    ``lower_dual_gap`` = ``lower_hedge.hedge_cost - lower_bound`` should be
    zero to solver tolerance — the LP has no duality gap (BHP 2013, Thm 1).
    """

    upper_bound: float
    lower_bound: float
    upper_hedge: MOTHedgePortfolio
    lower_hedge: MOTHedgePortfolio
    upper_dual_gap: float
    lower_dual_gap: float
    forward: float
    n_strikes: int
    n_grid: int


# ---------------------------------------------------------------------------
# Black–Scholes helpers (SYNTHETIC data generation only)
# ---------------------------------------------------------------------------


def _bs_undiscounted_call(
    spot: float, strike: float, maturity: float, rate: float, sigma: float
) -> float:
    """Undiscounted expected BS call payoff: E[(S_T - K)^+] = e^{rT} * C_BS."""
    if maturity <= 0 or sigma <= 0 or spot <= 0 or rate < 0:
        raise ValueError("maturity, sigma, spot must be > 0; rate >= 0")
    vol = sigma * np.sqrt(maturity)
    d1 = (np.log(spot / strike) + (rate + 0.5 * sigma**2) * maturity) / vol
    d2 = d1 - vol
    return float(spot * np.exp(rate * maturity) * norm.cdf(d1) - strike * norm.cdf(d2))


def generate_synthetic_vanillas(
    spot: float = 100.0,
    rate: float = 0.0,
    maturity: float = 0.5,
    sigma: float = 0.2,
    n_strikes: int = 21,
    strike_width: float = 0.3,
    seed: int | None = None,
) -> dict[str, Array | float]:
    """Generate SYNTHETIC vanilla call prices from the Black–Scholes model.

    Returns a dict with keys ``strikes`` (float array), ``call_prices``
    (undiscounted expected payoff array), ``forward`` (float), ``rate``,
    ``maturity``, and ``sigma``.  This is a **correctness** data source:
    prices are computed from a *known* model, then the MOT solver must
    recover bounds that contain the true model price.  All data is labelled
    SYNTHETIC — never market evidence.

    Parameters
    ----------
    spot : initial asset price.
    rate : risk-free rate (continuous compounding).
    maturity : time to expiration in years.
    sigma : Black–Scholes volatility.
    n_strikes : number of strike levels, centred on the forward.
    strike_width : half-width of the strike range relative to the forward
        (e.g. 0.3 → strikes from F*(1-0.3) to F*(1+0.3)).
    seed : ignored (reserved for future stochastic strike selection).
    """
    _ = seed  # deterministic by construction
    fwd = spot * np.exp(rate * maturity)
    low = fwd * (1.0 - strike_width)
    high = fwd * (1.0 + strike_width)
    strikes_arr = np.linspace(low, high, int(n_strikes), dtype=float)
    prices = np.array(
        [_bs_undiscounted_call(spot, float(k), maturity, rate, sigma) for k in strikes_arr],
        dtype=float,
    )
    return {
        "strikes": strikes_arr,
        "call_prices": prices,
        "forward": float(fwd),
        "rate": float(rate),
        "maturity": float(maturity),
        "sigma": float(sigma),
    }


# ---------------------------------------------------------------------------
# Constraint-matrix builder
# ---------------------------------------------------------------------------


def _build_mot_matrix(
    state_grid: Array,
    strikes: Array,
    forward: float,
) -> Array:
    """Build the equality constraint matrix A_eq for the MOT LP.

    The LP variables are π_j >= 0 (transport plan weights on state_grid[j]).
    The RHS b_eq = [1.0, forward, call_prices] is assembled by the caller.

    Returns ``A_eq`` where
        A_eq[0, :] = 1.0                  (sum to 1)
        A_eq[1, :] = state_grid           (martingale: E[S_T] = forward)
        A_eq[2:, :] = (grid - K_i)^+      (call price constraints)

    and
        b_eq = [1.0, forward, call_prices].
    """
    n_grid = int(state_grid.size)
    n_strikes = int(strikes.size)

    A_eq = np.empty((2 + n_strikes, n_grid), dtype=float)
    A_eq[0, :] = 1.0
    A_eq[1, :] = state_grid

    for i in range(n_strikes):
        A_eq[2 + i, :] = np.maximum(state_grid - float(strikes[i]), 0.0)

    return A_eq


def _validate_inputs(
    state_grid: Array,
    exotic_payoff: Array,
    strikes: Array,
    call_prices: Array,
    forward: float,
) -> tuple[Array, Array, Array, Array, float, int, int]:
    """Validate and coerce MOT inputs. Fail-closed on any violation."""
    x = np.asarray(state_grid, dtype=float).ravel()
    f = np.asarray(exotic_payoff, dtype=float).ravel()
    if x.size < 3:
        raise ValueError("state_grid must have at least 3 points")
    if x.size != f.size:
        raise ValueError(f"state_grid ({x.size}) and exotic_payoff ({f.size}) must match length")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(f)):
        raise ValueError("state_grid and exotic_payoff must be finite")
    if not np.all(np.diff(x) > 0):
        raise ValueError("state_grid must be strictly increasing")

    K = np.asarray(strikes, dtype=float).ravel()
    C = np.asarray(call_prices, dtype=float).ravel()
    if K.size < 1:
        raise ValueError("strikes must be non-empty")
    if K.size != C.size:
        raise ValueError(f"strikes ({K.size}) and call_prices ({C.size}) must match length")
    if not np.all(np.isfinite(K)) or not np.all(np.isfinite(C)):
        raise ValueError("strikes and call_prices must be finite")
    if not np.all(np.diff(K) > 0):
        raise ValueError("strikes must be strictly increasing")
    if np.any(C < 0):
        raise ValueError("call_prices must be non-negative")

    if not np.isfinite(forward) or forward <= 0:
        raise ValueError("forward must be finite and positive")

    # Sanity: call prices should be decreasing in strike and convex (no
    # butterfly arbitrage).  We only warn through the LP infeasibility
    # path, but we do check the domain basics.
    return x, f, K, C, float(forward), int(x.size), int(K.size)


def _check_martingale_feasibility(
    x: Array, K: Array, C: Array, forward: float, n: int, m: int
) -> None:
    """Detect a grossly impossible marginal (e.g. forward incompatible with
    bounded support). This is a fast pre-check; actual infeasibility is
    caught by the LP solver."""
    # If the grid spans from x[0] to x[-1], the martingale constraint
    # requires x[0] <= forward <= x[-1].
    if forward < x[0] - 1e-10 or forward > x[-1] + 1e-10:
        raise ValueError(f"forward {forward} outside state_grid range [{x[0]:.4f}, {x[-1]:.4f}]")
    # Call price C_i = E[(S_T - K_i)^+]; feasible range is
    # [max(x[0] - K_i, 0), max(x[-1] - K_i, 0)] where x[0] and x[-1]
    # are the smallest and largest grid points.
    for i in range(m):
        lo = float(max(x[0] - K[i], 0.0))
        hi = float(max(x[-1] - K[i], 0.0))
        if C[i] < lo - 1e-10 or C[i] > hi + 1e-10:
            raise ValueError(
                f"call_price[{i}] = {C[i]:.6f} is outside feasible range "
                f"[{lo:.6f}, {hi:.6f}] for strike {K[i]:.4f} "
                f"given grid [{x[0]:.4f}, {x[-1]:.4f}]"
            )


# ---------------------------------------------------------------------------
# Core LP solvers
# ---------------------------------------------------------------------------


def _linprog_mot(
    c: Array,
    A_eq: Array,
    b_eq: Array,
    bound_label: Literal["upper", "lower"],
) -> tuple[float, MOTHedgePortfolio, float]:
    """Solve the MOT LP via scipy's HiGHS and unpack dual variables.

    ``c`` is the exotic payoff vector (negated for upper bound, since
    linprog always minimises).  ``A_eq @ π = b_eq`` with ``π >= 0``.

    Returns ``(primal_value, hedge_portfolio, dual_gap)``.
    """
    n = int(c.size)
    result = linprog(
        c,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=[(0.0, None)] * n,
        method="highs",
        options={"primal_feasibility_tolerance": 1e-9, "dual_feasibility_tolerance": 1e-9},
    )

    if not result.success:
        raise RuntimeError(
            f"MOT {bound_label}-bound LP failed: status {result.status}, message: {result.message}"
        )

    primal_value = float(result.fun)
    if bound_label == "upper":
        # linprog solved min -f·x; primal (max) = -result.fun
        primal_value = -primal_value

    # Dual variables.  linprog returns λ such that for min c·x s.t.
    # A·x = b, b·λ = c·x*.  The MOT dual for upper bound has variables
    # y with A^T·y >= f; the relationship is y = -λ when we negated c.
    lam = np.asarray(result.eqlin.marginals, dtype=float)
    if bound_label == "upper":
        y = -lam  # sign flip for max → min conversion
    else:
        y = lam

    # y = [ν, η, α_1, ..., α_M]
    cash = float(y[0])
    forward_pos = float(y[1])
    option_pos = y[2:].copy()

    hedge_cost = cash + forward_pos * float(b_eq[1]) + float(np.dot(option_pos, b_eq[2:]))
    dual_gap = abs(primal_value - hedge_cost)

    scale = max(abs(primal_value), 1.0)
    if dual_gap > _DUAL_TOL * scale:
        raise RuntimeError(
            f"MOT {bound_label}-bound strong duality violated: "
            f"primal={primal_value:.8f}, dual={hedge_cost:.8f}, "
            f"gap={dual_gap:.2e}"
        )

    hedge = MOTHedgePortfolio(
        cash=cash,
        forward_position=forward_pos,
        option_positions=option_pos,
        hedge_cost=hedge_cost,
    )
    return primal_value, hedge, dual_gap


def solve_mot_upper_bound(
    state_grid: Array,
    exotic_payoff: Array,
    strikes: Array,
    call_prices: Array,
    forward: float,
) -> tuple[float, MOTHedgePortfolio]:
    """Solve the MOT upper bound (max expected exotic payoff).

    Parameters
    ----------
    state_grid : 1-D strictly increasing array of terminal spot states.
    exotic_payoff : exotic option payoff at each state_grid point.
    strikes : market strike prices (strictly increasing).
    call_prices : undiscounted expected call payoffs (market price × e^{rT}).
    forward : forward price = spot × exp(rate × maturity).

    Returns
    -------
    (upper_bound, hedge) : the maximised expected payoff and the cheapest
        superhedging portfolio.  hedge.hedge_cost should equal upper_bound
        (strong duality).

    Raises
    ------
    ValueError : invalid or infeasible inputs.
    RuntimeError : LP solver failure or duality gap.
    """
    x, f, K, C, F, n, m = _validate_inputs(state_grid, exotic_payoff, strikes, call_prices, forward)
    _check_martingale_feasibility(x, K, C, F, n, m)
    A_eq = _build_mot_matrix(x, K, F)
    b_eq = np.concatenate([[1.0, F], C])

    primal, hedge, _ = _linprog_mot(-f, A_eq, b_eq, "upper")
    return primal, hedge


def solve_mot_lower_bound(
    state_grid: Array,
    exotic_payoff: Array,
    strikes: Array,
    call_prices: Array,
    forward: float,
) -> tuple[float, MOTHedgePortfolio]:
    """Solve the MOT lower bound (min expected exotic payoff).

    Parameters are the same as :func:`solve_mot_upper_bound`.

    Returns
    -------
    (lower_bound, hedge) : the minimised expected payoff and the most
        expensive subhedging portfolio.
    """
    x, f, K, C, F, n, m = _validate_inputs(state_grid, exotic_payoff, strikes, call_prices, forward)
    _check_martingale_feasibility(x, K, C, F, n, m)
    A_eq = _build_mot_matrix(x, K, F)
    b_eq = np.concatenate([[1.0, F], C])

    primal, hedge, _ = _linprog_mot(f, A_eq, b_eq, "lower")
    return primal, hedge


def solve_mot_bounds(
    state_grid: Array,
    exotic_payoff: Array,
    strikes: Array,
    call_prices: Array,
    forward: float,
) -> MOTBounds:
    """Solve both upper and lower MOT bounds for an exotic option.

    Parameters as in :func:`solve_mot_upper_bound`.

    Returns
    -------
    MOTBounds with upper/lower primal values, dual hedge portfolios,
    and dual gaps (the difference between primal and dual values).
    """
    x, f, K, C, F, n, m = _validate_inputs(state_grid, exotic_payoff, strikes, call_prices, forward)
    _check_martingale_feasibility(x, K, C, F, n, m)
    A_eq = _build_mot_matrix(x, K, F)
    b_eq = np.concatenate([[1.0, F], C])

    ub_val, ub_hedge, ub_gap = _linprog_mot(-f, A_eq, b_eq, "upper")
    lb_val, lb_hedge, lb_gap = _linprog_mot(f, A_eq, b_eq, "lower")

    return MOTBounds(
        upper_bound=ub_val,
        lower_bound=lb_val,
        upper_hedge=ub_hedge,
        lower_hedge=lb_hedge,
        upper_dual_gap=ub_gap,
        lower_dual_gap=lb_gap,
        forward=float(F),
        n_strikes=m,
        n_grid=n,
    )


# ---------------------------------------------------------------------------
# Canonical application: variance swap (log-contract replication)
# ---------------------------------------------------------------------------


def variance_swap_mot_bounds(
    spot: float = 100.0,
    rate: float = 0.0,
    maturity: float = 0.5,
    sigma: float = 0.2,
    n_strikes: int = 21,
    n_grid: int = 201,
    grid_width: float = 0.6,
    strike_width: float = 0.3,
) -> dict:
    """MOT bounds for a variance swap fair strike versus BS-synthetic vanillas.

    Generates SYNTHETIC call prices from a Black–Scholes model, then
    *forgets* the model and solves the MOT LP for the log-contract payoff

        f(S_T) = (2/T) * (-ln(S_T / F)),

    whose undiscounted expectation equals the fair variance strike. The
    classical Neuberger (1990/1994) log-contract replication formula should
    emerge in the dual: the optimal static option positions α_i approximate
    the weight function (2/T) * (1/K_i^2).

    Returns a dict with the MOT upper/lower bounds on the fair variance
    strike, the BS-theoretic value, the spread (as fraction of the BS
    value), the dual gaps, and the optimal static hedge weights.
    All data is SYNTHETIC — correctness evidence, never market evidence.
    """
    van = generate_synthetic_vanillas(
        spot=spot,
        rate=rate,
        maturity=maturity,
        sigma=sigma,
        n_strikes=n_strikes,
        strike_width=strike_width,
    )
    F = float(van["forward"])
    K_arr = np.asarray(van["strikes"], dtype=float)
    C_arr = np.asarray(van["call_prices"], dtype=float)
    mat = float(van["maturity"])

    # State grid
    low = F * (1.0 - grid_width)
    high = F * (1.0 + grid_width)
    x = np.linspace(low, high, int(n_grid), dtype=float)

    # Log-contract payoff: (2/T) * (-ln(S_T/F))
    with np.errstate(divide="ignore"):
        f = (2.0 / mat) * (-np.log(np.maximum(x, _EPS) / F))

    result = solve_mot_bounds(x, f, K_arr, C_arr, F)

    # BS-theoretic fair variance: E[(2/T)*(-ln(S_T/F))] = sigma^2
    # Under BS: ln(S_T/F) = -0.5*sigma^2*T + sigma*W_T
    # E[-ln(S_T/F)] = 0.5*sigma^2*T
    # So f_expected = (2/T) * (0.5*sigma^2*T) = sigma^2
    bs_fair_variance = float(sigma**2)

    spread = (result.upper_bound - result.lower_bound) / max(abs(bs_fair_variance), 1e-12)

    # Theoretical log-contract weights: w(K) = (2/T) * (1/K^2) * dK
    # The dual option positions α_i approximate this weight.
    dK = float(np.mean(np.diff(K_arr)))
    theo_weights = (2.0 / mat) * (1.0 / K_arr**2) * dK

    return {
        "mot_upper_bound": result.upper_bound,
        "mot_lower_bound": result.lower_bound,
        "bs_fair_variance": bs_fair_variance,
        "bs_contained": bool(result.lower_bound <= bs_fair_variance <= result.upper_bound),
        "spread": float(spread),
        "upper_dual_gap": result.upper_dual_gap,
        "lower_dual_gap": result.lower_dual_gap,
        "hedge_cash": result.upper_hedge.cash,
        "hedge_forward": result.upper_hedge.forward_position,
        "hedge_options": result.upper_hedge.option_positions,
        "theo_weights": theo_weights,
        "strikes": K_arr,
        "forward": F,
        "maturity": mat,
    }
