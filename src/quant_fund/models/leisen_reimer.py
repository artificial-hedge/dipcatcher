"""Leisen-Reimer (1996) binomial option tree (SYNTHETIC).

The Leisen-Reimer tree chooses up/down moves and the risk-neutral probability by
inverting the Black-Scholes ``d1``/``d2`` through a Peizer-Pratt normal
approximation, so the tree centres on the strike and converges much faster
(order 1/n^2, monotonically) than the Cox-Ross-Rubinstein tree.  The step count
``n`` must be odd.

Reference: D. Leisen, M. Reimer (1996), "Binomial models for option valuation --
examining and improving convergence", Applied Mathematical Finance.  Fail-closed
on invalid parameters.
"""

from __future__ import annotations

import numpy as np


def _peizer_pratt(z: float, n: int) -> float:
    """Peizer-Pratt method-2 inversion of the normal CDF onto (0, 1)."""
    c = z / (n + 1.0 / 3.0 + 0.1 / (n + 1.0))
    return float(0.5 + np.sign(z) * 0.5 * np.sqrt(1.0 - np.exp(-(c**2) * (n + 1.0 / 6.0))))


def leisen_reimer(
    s: float,
    k: float,
    t: float,
    r: float,
    q: float,
    sigma: float,
    n: int = 101,
    option: str = "call",
    american: bool = False,
) -> float:
    """Leisen-Reimer binomial price (European or American)."""
    if s <= 0 or k <= 0 or t <= 0 or sigma <= 0:
        raise ValueError("require positive spot, strike, maturity and sigma")
    if option not in {"call", "put"}:
        raise ValueError("option must be 'call' or 'put'")
    if n < 3:
        raise ValueError("n must be >= 3")
    if n % 2 == 0:
        n += 1  # Leisen-Reimer requires an odd number of steps
    dt = t / n
    d1 = (np.log(s / k) + (r - q + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    p = _peizer_pratt(d2, n)
    p_prime = _peizer_pratt(d1, n)
    growth = np.exp((r - q) * dt)
    u = growth * p_prime / p
    d = (growth - p * u) / (1.0 - p)
    disc = np.exp(-r * dt)
    j = np.arange(n + 1, dtype=float)
    spot = s * u ** (n - j) * d**j
    call = option == "call"
    value = np.maximum(spot - k, 0.0) if call else np.maximum(k - spot, 0.0)
    for i in range(n - 1, -1, -1):
        value = disc * (p * value[: i + 1] + (1.0 - p) * value[1 : i + 2])
        if american:
            j_i = np.arange(i + 1, dtype=float)
            spot_i = s * u ** (i - j_i) * d**j_i
            intrinsic = np.maximum(spot_i - k, 0.0) if call else np.maximum(k - spot_i, 0.0)
            value = np.maximum(value, intrinsic)
    return float(value[0])


def bench_leisen_reimer(seed: int = 20261231 + 972) -> dict[str, float]:
    """Leisen-Reimer vs BSM oracle for Europeans, plus American floor
    and put-call parity."""
    from scipy.stats import norm

    rng = np.random.default_rng(seed)
    err = parity = amer_gap = 0.0
    for _ in range(12):
        s = float(rng.uniform(60, 140))
        k = float(s * rng.uniform(0.85, 1.15))
        t = float(rng.uniform(0.2, 1.5))
        r = float(rng.uniform(0.0, 0.05))
        sig = float(rng.uniform(0.1, 0.5))
        c_lr = leisen_reimer(s, k, t, r, 0.0, sig, n=201, option="call")
        p_lr = leisen_reimer(s, k, t, r, 0.0, sig, n=201, option="put")
        d1 = (np.log(s / k) + (r + 0.5 * sig * sig) * t) / (sig * np.sqrt(t))
        d2 = d1 - sig * np.sqrt(t)
        c_bs = s * norm.cdf(d1) - k * np.exp(-r * t) * norm.cdf(d2)
        p_bs = c_bs - s + k * np.exp(-r * t)
        err = max(err, abs(c_lr - c_bs), abs(p_lr - p_bs))
        parity = max(parity, abs(c_lr - p_lr - (s - k * np.exp(-r * t))))
        a_lr = leisen_reimer(s, k, t, r, 0.0, sig, n=101, option="put", american=True)
        amer_gap = max(amer_gap, p_lr - a_lr)  # American < European would be a defect
    if err > 0.02:
        raise ValueError(f"Leisen-Reimer off BSM oracle: {err}")
    if parity > 1e-8:
        raise ValueError(f"put-call parity violated: {parity}")
    if amer_gap > 1e-9:
        raise ValueError(f"American put priced below European: {amer_gap}")
    return {
        "synthetic_lr_bsm_err": err,
        "synthetic_lr_parity_err": parity,
        "synthetic_lr_amer_below_eur": amer_gap,
    }
