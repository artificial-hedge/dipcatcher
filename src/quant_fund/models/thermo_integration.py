"""Thermodynamic integration: free energy along a lambda path (SYNTHETIC).

F(1)-F(0) = ∫_0^1 <dU/dλ>_λ dλ interpolating a flat base U0=x^4 and
double well U1 = x^4 - 4x²; MC at each λ on a grid, trapezoid
estimate vs the quadrature answer.
"""

import numpy as np


def _u0(x: np.ndarray) -> np.ndarray:
    return x**4


def _u1(x: np.ndarray) -> np.ndarray:
    return x**4 - 4.0 * x**2


def _mc_mean(lam: float, n: int, rng: np.random.Generator) -> float:
    x = 0.5
    tot = 0.0
    for _ in range(n):
        prop = x + rng.normal(0, 0.4)
        du = (1 - lam) * (_u0(np.array(prop)) - _u0(np.array(x))) + lam * (
            _u1(np.array(prop)) - _u1(np.array(x))
        )
        if rng.random() < np.exp(-float(du)):
            x = prop
        tot += float(_u1(np.array(x)) - _u0(np.array(x)))
    return tot / n


def bench_thermo_integration(seed: int = 5611) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    lams = np.linspace(0, 1, 9)
    means = np.array([_mc_mean(lam, 4000, rng) for lam in lams])
    df_est = float(np.trapezoid(means, lams))
    grid = np.linspace(-4, 4, 40001)
    z0 = np.trapezoid(np.exp(-_u0(grid)), grid)
    z1 = np.trapezoid(np.exp(-_u1(grid)), grid)
    df_true = float(-np.log(z1) + np.log(z0))
    return {
        "synthetic_ti_df": df_est,
        "synthetic_ti_truth": df_true,
        "synthetic_ti_err": abs(df_est - df_true),
    }
