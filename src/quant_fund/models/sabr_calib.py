"""SABR calibration: fit (alpha, beta, rho, nu) to a synthetic smile
by matching Hagan lognormal-IV quotes; bench recovers parameters on
a known-parameter surface via least squares on IV.
"""

import numpy as np
from scipy.optimize import least_squares


def _hagan_iv(
    k: float, f: float, t: float, alpha: float, beta: float, rho: float, nu: float
) -> float:
    if abs(k - f) < 1e-8:
        fk = f ** (1 - beta)
        term1 = alpha / fk
        return float(
            term1
            * (
                1
                + t
                * (
                    (1 - beta) ** 2 * alpha**2 / (24 * fk**2)
                    + rho * beta * nu * alpha / (4 * fk)
                    + (2 - 3 * rho**2) * nu**2 / 24
                )
            )
        )
    fk = (f * k) ** ((1 - beta) / 2)
    fk2 = (f * k) ** (1 - beta)
    z = (nu / alpha) * fk * np.log(f / k)
    xz = np.log((np.sqrt(1 - 2 * rho * z + z * z) + z - rho) / (1 - rho))
    zz = z / xz if abs(z) > 1e-10 else 1.0
    return float(
        alpha
        / (
            fk
            * (
                1
                + (1 - beta) ** 2 / 24 * np.log(f / k) ** 2
                + (1 - beta) ** 4 / 1920 * np.log(f / k) ** 4
            )
        )
        * zz
        * (
            1
            + t
            * (
                (1 - beta) ** 2 * alpha**2 / (24 * fk2)
                + rho * beta * nu * alpha / (4 * (f * k) ** ((1 - beta) / 2))
                + (2 - 3 * rho**2) * nu**2 / 24
            )
        )
    )


def bench_sabr_calib(seed: int = 5803) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    f, t = 1.0, 0.5
    true = np.array([0.25, 0.5, -0.3, 0.6])  # alpha, beta, rho, nu
    ks = np.linspace(0.7, 1.3, 9)
    ivs = np.array([_hagan_iv(k, f, t, *true) for k in ks])
    ivs += rng.normal(0, 0.0005, len(ks))

    def resid(p: np.ndarray) -> np.ndarray:
        a, b, r, v = p[0], np.clip(p[1], 0, 1), np.clip(p[2], -0.99, 0.99), np.exp(p[3])
        r_arr: np.ndarray = np.asarray(
            [_hagan_iv(k, f, t, a, b, r, v) for k in ks], dtype=float
        ) - np.asarray(ivs, dtype=float)
        return r_arr

    res = least_squares(resid, np.array([0.2, 0.5, 0.0, -0.5]), method="lm")
    a, b, r, v = res.x[0], np.clip(res.x[1], 0, 1), np.clip(res.x[2], -0.99, 0.99), np.exp(res.x[3])
    est = np.array([a, b, r, v])
    err = np.abs(est - true)
    return {
        "synthetic_sabr_alpha_err": float(err[0]),
        "synthetic_sabr_rho_err": float(err[2]),
        "synthetic_sabr_nu_err": float(err[3]),
        "synthetic_sabr_iv_rmse": float(np.sqrt(np.mean(resid(res.x) ** 2))),
    }
