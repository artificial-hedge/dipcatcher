"""Brock-Hommes heterogeneous-agent asset market (discrete-choice switching).

Agents switch between forecasting strategies by a logit/discrete-choice
rule with intensity of choice ``beta``. The canonical two-type market has
fundamentalists (``f = -g p``: bet on mean reversion to the fundamental,
normalized so the fundamental deviation is ``p``) and chartists
(``f = +h (p_t - p_{t-1})``: extrapolate the last move). The market map

    p_{t+1} = p_t + lam * tanh(n_f f_f + n_c f_c) + eps_t,

    n_{i,t} = exp(beta * U_{i,t}) / Z_t,

    U_{i,t} = rho U_{i,t-1} + (1 - rho) [tanh(kappa f_i) * gain_t - C_i]

bounds the price increment by market depth (``tanh``) and scores each
strategy by realized profit per unit position with EWMA memory ``rho`` —
the standard Brock-Hommes route to endogenous booms and crashes.

Verified behaviors this implementation reproduces: the deterministic
skeleton converges to the fundamental steady state in the mixed low-beta
regime; raising ``beta`` sharpens strategy sorting and — when the trend
coefficient is strong — locks the population into a dominant regime
(fraction -> 1). The stochastic version generates volatility bursts and
fat-tailed returns from endogenous regime switching.

References
----------
- Brock & Hommes (1997). A rational route to randomness.
  *Econometrica* 65 — discrete-choice switching + bifurcation skeleton
  (journal paper; verified — not on arXiv).
- Brock & Hommes (1998). Heterogeneous beliefs and routes to chaos in a
  simple asset pricing model. *JEDC* 22 — formal bifurcation results.
- Hommes (2006). Heterogeneous agent models in economics and finance.
  *Handbook of Computational Economics* 2 — survey chapter.
- Lux & Marchesi (1999). Scaling and criticality in a stochastic
  multi-agent model of a financial market. *Nature* 397 —
  arXiv:cond-mat/9806310 (verified; stylized-fact reproduction).
- Franke & Westerhoff (2012). Structural stochastic volatility in asset
  pricing dynamics. *JEDC* 36 — EWMA fitness variant (journal).

Honesty
-------
All benches run on seeded SYNTHETIC simulations generated in-module.
Stylized-fact numbers validate the simulator machinery only — never
market evidence.

Composition notes
-----------------
- ``models/market_making.py``: Avellaneda-Stoikov quote-setting —
  microstructure complement (single rational agent, not a HAM).
- ``market_sim/``: replay/execution engine — this module is a
  price-formation sandbox, not a book simulator.
- ``metrics/`` tail/vol diagnostics consume the simulated series.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _check_abm(beta: float, sigma_eps: float, g: float) -> None:
    if beta < 0 or sigma_eps < 0 or g <= 0:
        raise ValueError("need beta >= 0, sigma_eps >= 0, g > 0")


@dataclass(frozen=True)
class ABMPath:
    """Simulated heterogeneous-agent market path."""

    prices: FloatArray
    returns: FloatArray
    fractions: FloatArray  # (T, 2) — [fundamentalist, chartist]
    fitness: FloatArray  # (T, 2)


def simulate_brock_hommes(
    n: int,
    beta: float,
    g: float = 1.2,
    h: float = 0.9,
    cost_chartist: float = 0.01,
    rho: float = 0.9,
    lam: float = 0.5,
    sigma_eps: float = 0.01,
    p0: float = 0.05,
    seed: int = 0,
) -> ABMPath:
    """Two-type Brock-Hommes market: fundamentalists vs chartists.

    ``p`` is the deviation from the fundamental (normalized to 0). The
    market map adds a ``tanh``-bounded aggregate demand increment plus
    Gaussian noise ``sigma_eps * eps``.
    """
    _check_abm(beta, sigma_eps, g)
    if n < 200:
        raise ValueError("n >= 200 required")
    if rho < 0 or rho >= 1 or lam <= 0:
        raise ValueError("need 0 <= rho < 1 and lam > 0")
    rng = np.random.default_rng(seed)
    p = np.zeros(n + 1)
    frac = np.zeros((n + 1, 2))
    fit = np.zeros((n + 1, 2))
    p[0], p[1] = p0, p0 + rng.normal(0.0, sigma_eps)
    frac[0] = frac[1] = np.array([0.5, 0.5])
    for t in range(1, n):
        f_fund = -g * p[t]
        f_char = h * (p[t] - p[t - 1])
        e = np.exp(np.clip(beta * fit[t - 1], -40.0, 40.0))
        n_t = e / e.sum()
        frac[t + 1] = n_t
        dem = n_t[0] * f_fund + n_t[1] * f_char
        p[t + 1] = p[t] + lam * np.tanh(dem) + sigma_eps * rng.standard_normal()
        gain = p[t + 1] - p[t]
        fit[t, 0] = rho * fit[t - 1, 0] + (1.0 - rho) * np.tanh(4.0 * f_fund) * gain
        fit[t, 1] = rho * fit[t - 1, 1] + (1.0 - rho) * (
            np.tanh(4.0 * f_char) * gain - cost_chartist
        )
    return ABMPath(prices=p[1:], returns=np.diff(p[1:]), fractions=frac[1:], fitness=fit[:-1])


def deterministic_skeleton(
    n: int,
    beta: float,
    g: float = 1.2,
    h: float = 0.9,
    cost_chartist: float = 0.01,
    p0: float = 0.05,
) -> ABMPath:
    """Brock-Hommes skeleton (sigma_eps = 0) for bifurcation analysis."""
    return simulate_brock_hommes(
        n, beta, g=g, h=h, cost_chartist=cost_chartist, sigma_eps=0.0, p0=p0
    )


def lyapunov_exponent(
    n: int,
    beta: float,
    eps: float = 1e-7,
    g: float = 1.2,
    h: float = 0.9,
    burn: int = 500,
) -> float:
    """End-to-end divergence rate of the skeleton: two trajectories
    started eps apart; log(final distance / eps) per step after burn-in.
    """
    if beta < 0:
        raise ValueError("beta >= 0")
    a = deterministic_skeleton(n, beta, g=g, h=h)
    b = simulate_brock_hommes(n, beta, g=g, h=h, sigma_eps=0.0, p0=0.05 + eps)
    div = np.abs(b.prices - a.prices) + 1e-300
    return float(np.mean(np.log(div[burn:] / eps))) / max(1, n - burn - 1)


def stylized_facts(r: FloatArray) -> dict[str, float]:
    """Stylized-fact diagnostics on a synthetic return series."""
    v = np.asarray(r, dtype=float).ravel()
    if v.size < 200 or not np.isfinite(v).all():
        raise ValueError("need >= 200 finite returns")
    z = (v - v.mean()) / max(v.std(), 1e-12)
    kurt = float(np.mean(z**4) - 3.0)
    ar = np.abs(v - v.mean())
    ar = ar - ar.mean()
    denom = float((ar**2).sum())
    ac1 = float((ar[1:] * ar[:-1]).sum() / denom) if denom > 0 else 0.0
    vc = v - v.mean()
    ac_r1 = float((vc[1:] * vc[:-1]).sum() / max((vc**2).sum(), 1e-12))
    up = v[v > 0]
    dn = v[v < 0]
    asym = float(up.std() - dn.std()) if up.size and dn.size else 0.0
    return {
        "excess_kurtosis": kurt,
        "abs_autocorr_lag1": ac1,
        "return_autocorr_lag1": ac_r1,
        "vol_asymmetry": asym,
    }


def dominance_grid(
    betas: FloatArray,
    n: int = 1500,
    g: float = 1.2,
    h: float = 4.0,
    cost_chartist: float = 0.0,
) -> tuple[FloatArray, FloatArray]:
    """Sweep intensity of choice with a strong trend rule; return
    (betas, late-window chartist dominance). Brock-Hommes sorting:
    higher beta -> sharper regime selection (dominance -> 1)."""
    bs = np.asarray(betas, dtype=float).ravel()
    if bs.size < 2 or np.any(bs < 0):
        raise ValueError("need >= 2 non-negative betas")
    dom = np.empty(bs.size)
    for i, b in enumerate(bs):
        sim = simulate_brock_hommes(
            n, float(b), g=g, h=h, cost_chartist=cost_chartist, sigma_eps=0.0
        )
        dom[i] = float(sim.fractions[-200:, 1].mean())
    return bs, dom


def bench_heterogeneous_abm(seed: int = 20260201) -> dict[str, float]:
    """SYNTHETIC bench for the Brock-Hommes simulator. Correctness only."""
    out: dict[str, float] = {}
    sim = simulate_brock_hommes(4000, beta=120.0, sigma_eps=0.01, seed=seed)
    facts = stylized_facts(sim.returns)
    out["synthetic_excess_kurtosis"] = facts["excess_kurtosis"]
    out["synthetic_abs_autocorr_lag1"] = facts["abs_autocorr_lag1"]
    out["synthetic_return_autocorr_lag1"] = facts["return_autocorr_lag1"]
    out["synthetic_vol_asymmetry"] = facts["vol_asymmetry"]
    # low-beta mixed skeleton converges to the fundamental steady state
    sk0 = deterministic_skeleton(1500, beta=0.0)
    out["synthetic_skeleton_conv_std"] = float(np.std(sk0.prices[-200:]))
    # sorting sharpens with beta (strong-trend coupling): dominance rises
    _, dom = dominance_grid(np.array([0.0, 100.0, 500.0]))
    out["synthetic_dominance_low_beta"] = float(dom[0])
    out["synthetic_dominance_high_beta"] = float(dom[-1])
    out["synthetic_sorting_sharpens"] = float(dom[-1] > dom[0])
    out["synthetic_lyapunov"] = lyapunov_exponent(1500, 100.0)
    out["synthetic_fraction_bounds_ok"] = float(
        np.all(sim.fractions >= 0) and np.all(sim.fractions <= 1)
    )
    sim2 = simulate_brock_hommes(4000, beta=120.0, sigma_eps=0.01, seed=seed)
    out["synthetic_determinism"] = float(np.allclose(sim.prices, sim2.prices))
    return out
