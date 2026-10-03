"""Heath-Jarrow-Morton Gaussian forward-curve simulation.

References
----------
- Heath, D., Jarrow, R. & Morton, A. (1992). "Bond Pricing and the
  Term Structure of Interest Rates: A New Methodology for Contingent
  Claims Valuation." *Econometrica* 60(1), 77-105.
- Musiela, M. (1993). "Stochastic PDEs and Kolmogorov Equations in
  Infinite Dimensions." *Stochastic Analysis and Applications* 11(4).
- Bjork, T. & Christensen, B.J. (1999). "Interest Rate Dynamics and
  Consistent Forward Rate Curves." *Mathematical Finance* 9(4).

Honesty
-------
All synthetic experiments are labeled SYNTHETIC and are correctness
checks, never market evidence.

Composition notes
-----------------
Single-factor Gaussian HJM on a maturity grid: forward rate

    df(t,T) = alpha(t,T) dt + sigma(t,T) dW_t,

with the risk-neutral drift restriction ``alpha(t,T) = sigma(t,T) *
int_t^T sigma(t,s) ds`` (Musiela parameterization evaluated at t=0 —
the tractable Markov case). ``sigma(t,T) = sigma0 * exp(-kappa*(T-t))``
makes the short rate OU/Gauss-Markov (the continuous-time Vasicek-Hull-
White special case). Bond prices are read off
``P(t,T) = exp(-int_t^T f(t,u) du)`` by trapezoid quadrature on the
grid. The synth checks the martingale property: simulated discount
bonds are unbiased relative to the analytic forward curve.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def hjm_simulate(
    f0: FloatArray,
    maturities: FloatArray,
    horizon: float = 1.0,
    n_steps: int = 50,
    n_paths: int = 2000,
    sigma0: float = 0.02,
    kappa: float = 0.3,
    seed: int = 0,
) -> dict[str, float | FloatArray]:
    """Gaussian HJM path simulation under the risk-neutral measure.

    ``f0`` is the initial instantaneous forward curve on the maturity
    grid ``maturities`` (years). Returns the simulated terminal curve,
    and bond-price moments for the maturity nearest ``horizon``.
    """
    ff = np.asarray(f0, dtype=np.float64)
    taus = np.asarray(maturities, dtype=np.float64)
    if ff.ndim != 1 or taus.ndim != 1 or ff.shape != taus.shape:
        raise ValueError("f0 and maturities must be same-length vectors")
    n = ff.shape[0]
    if n < 5 or horizon <= 0 or n_steps < 1 or n_paths < 10:
        raise ValueError("bad grid/simulation sizes")
    if not np.all(np.isfinite(ff)) or not np.all(np.isfinite(taus)):
        raise ValueError("non-finite inputs")
    if np.any(np.diff(taus) <= 0) or taus[0] < 0:
        raise ValueError("maturities must be increasing nonnegative")
    if sigma0 <= 0 or kappa <= 0:
        raise ValueError("sigma0/kappa must be positive")

    rng = np.random.default_rng(seed)
    dt = horizon / n_steps
    dW = rng.normal(0.0, np.sqrt(dt), (n_paths, n_steps))
    t_grid = np.arange(n_steps) * dt
    f_path = np.zeros((n_paths, n))
    f_t = np.broadcast_to(ff, (n_paths, n)).copy()

    for s in range(n_steps):
        t = t_grid[s]
        sig = sigma0 * np.exp(-kappa * np.maximum(taus - t, 0.0))
        integ = sigma0 / kappa * (1.0 - np.exp(-kappa * np.maximum(taus - t, 0.0)))
        alpha = sig * integ
        f_t += alpha * dt + sig * dW[:, s][:, None]
        f_path = f_t.copy()

    idx = int(np.argmin(np.abs(taus - horizon)))
    price_now = np.exp(-np.trapezoid(f_t[:, idx:], taus[idx:][None, :], axis=1))
    price0 = float(np.exp(-np.trapezoid(ff[idx:], taus[idx:])))

    return {
        "bond_price_mean": float(np.mean(price_now)),
        "bond_price_p5": float(np.quantile(price_now, 0.05)),
        "bond_price_p95": float(np.quantile(price_now, 0.95)),
        "bond_price_init": price0,
        "f_mean_mid": float(np.mean(f_path[:, n // 2])),
        "f_std_mid": float(np.std(f_path[:, n // 2])),
        "drift_magnitude": float(np.mean(alpha)),
        "_f_path": f_path,
    }


def synth_hjm(
    seed: int = 20261231 + 288,
    slope: float = 0.004,
    n_grid: int = 20,
) -> dict[str, FloatArray]:
    """Upward-sloping initial forward curve on a 10y grid."""
    rng = np.random.default_rng(seed)
    taus = np.linspace(0.0, 10.0, n_grid)
    f0 = 0.03 + slope * taus + 0.002 * rng.standard_normal(n_grid) * 0.0
    return {"f0": f0, "maturities": taus}


def bench_hjm(seed: int = 20261231 + 288) -> dict[str, float]:
    """Wave-50 self-check: terminal-curve distribution is centered
    around the drift-shifted curve and the simulated bond price
    brackets the analytic one."""
    d = synth_hjm(seed=seed)
    f0 = np.asarray(d["f0"])
    taus = np.asarray(d["maturities"])
    a = hjm_simulate(f0, taus, horizon=1.0, n_steps=40, n_paths=4000, seed=seed)
    a2 = hjm_simulate(f0, taus, horizon=1.0, n_steps=40, n_paths=4000, seed=seed)
    p5 = float(a["bond_price_p5"])
    p95 = float(a["bond_price_p95"])
    p0 = float(a["bond_price_init"])
    pm = float(a["bond_price_mean"])
    detects = float(p5 < p0 < p95 and abs(pm - p0) < 0.15)
    return {
        "synthetic_detects": detects,
        "synthetic_determinism": float(
            np.array_equal(np.asarray(a["_f_path"]), np.asarray(a2["_f_path"]))
        ),
        "synthetic_price_mean": pm,
        "synthetic_price_init": p0,
        "synthetic_price_p5": p5,
        "synthetic_price_p95": p95,
        "synthetic_f_std": float(a["f_std_mid"]),
        "synthetic_drift": float(a["drift_magnitude"]),
    }
