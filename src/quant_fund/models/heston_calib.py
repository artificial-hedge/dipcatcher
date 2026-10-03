"""Heston calibration: price calls by semi-analytic characteristic
function inversion (trapezoid integral), fit (v0, theta, kappa, xi,
rho) to synthetic quotes generated from known parameters, recover by
least squares.
"""

import numpy as np
from scipy.optimize import least_squares


def _cf(u: np.ndarray, s: float, k: float, t: float, p: np.ndarray) -> np.ndarray:
    v0, theta, kappa, xi, rho = p
    i = 1j
    d = np.sqrt((rho * xi * i * u - kappa) ** 2 + xi**2 * (i * u + u**2))
    g = (kappa - rho * xi * i * u - d) / (kappa - rho * xi * i * u + d)
    c = (
        kappa
        * theta
        / xi**2
        * ((kappa - rho * xi * i * u - d) * t - 2 * np.log((1 - g * np.exp(-d * t)) / (1 - g)))
    )
    dd = (kappa - rho * xi * i * u - d) / xi**2 * ((1 - np.exp(-d * t)) / (1 - g * np.exp(-d * t)))
    return np.asarray(np.exp(i * u * np.log(s) + c + dd * v0 - i * u * np.log(k)))


def _price(s: float, k: float, t: float, p: np.ndarray) -> float:
    u = np.linspace(1e-8, 60.0, 4000)
    inte = np.real(_cf(u - 0.5j, s, k, t, p) / (u**2 + 0.25))
    return float(s - np.sqrt(s * k) / np.pi * np.trapezoid(inte, u))


def bench_heston_calib(seed: int = 5807) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    s = 100.0
    # theta tied to v0 (stationary-mean convention) -> 4 free params,
    # two maturities to separate kappa
    true = np.array([0.04, 1.5, 0.4, -0.6])  # v0=theta, kappa, xi, rho
    ks = np.linspace(85, 115, 5)
    ts = np.array([0.25, 0.75])
    quotes = np.array(
        [[_price(s, k, tt, np.array([true[0], true[0], *true[1:]])) for k in ks] for tt in ts]
    ).ravel()
    quotes += rng.normal(0, 0.005, len(quotes))

    def _map(p: np.ndarray) -> np.ndarray:
        v0 = 0.01 + 0.09 / (1 + np.exp(-p[0]))
        kappa = 0.2 + 4.0 / (1 + np.exp(-p[1]))
        xi = 0.05 + 0.9 / (1 + np.exp(-p[2]))
        rho = np.tanh(p[3]) * 0.95
        return np.array([v0, v0, kappa, xi, rho])

    def resid(p: np.ndarray) -> np.ndarray:
        pp = _map(p)
        r_arr: np.ndarray = np.asarray(
            [_price(s, k, tt, pp) for tt in ts for k in ks], dtype=float
        ) - np.asarray(quotes, dtype=float)
        return r_arr

    res = least_squares(resid, np.zeros(4), method="lm", max_nfev=60)
    pp = _map(res.x)
    est = np.array([pp[0], pp[2], pp[3], pp[4]])
    err = np.abs(est - true)
    return {
        "synthetic_heston_v0_err": float(err[0]),
        "synthetic_heston_kappa_err": float(err[1]),
        "synthetic_heston_xi_err": float(err[2]),
        "synthetic_heston_rho_err": float(err[3]),
        "synthetic_heston_px_rmse": float(np.sqrt(np.mean(resid(res.x) ** 2))),
    }
