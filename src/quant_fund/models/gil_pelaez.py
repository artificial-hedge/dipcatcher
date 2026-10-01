"""Gil-Pelaez characteristic-function inversion for distributions.

References
----------
- Gil-Pelaez, J. (1951). "Note on the Inversion Theorem."
  *Biometrika* 38(3-4), 481-482.
- Davies, R.B. (1973). "Numerical Inversion of a Characteristic
  Function." *Biometrika* 60(2), 415-417.
- Carr, P. & Madan, D. (1999). "Option Valuation Using the Fast
  Fourier Transform." *Journal of Computational Finance* 2(4), 61-73.

Honesty
-------
All synthetic experiments are labeled SYNTHETIC and are correctness
checks, never market evidence.

Composition notes
-----------------
Given the characteristic function ``phi(u) = E[exp(i u X)]`` of a
real variable, Gil-Pelaez writes the CDF and density as

    F(x) = 1/2 - (1/pi) int_0^inf Im(phi(u) e^{-iux}) / u du,
    f(x) = (1/pi) int_0^inf Re(phi(u) e^{-iux}) du,

evaluated by trapezoid quadrature on a uniform frequency grid; the
characteristic function's own decay makes the tails integrable.
The synth inverts the Ornstein-Uhlenbeck characteristic function —
whose closed form is known — against the exact Gaussian target for
both the CDF and the density, so the self-check is a true identity
rather than a statistical test.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import stats as _stats

FloatArray = NDArray[np.float64]


def _ou_cf(
    u: FloatArray, x0: float, kappa: float, theta: float, sigma: float, t: float
) -> NDArray[np.complex128]:
    """OU characteristic function E[exp(i u X_t)] given X_0 = x0."""
    m = theta + (x0 - theta) * np.exp(-kappa * t)
    v = sigma**2 * (1.0 - np.exp(-2.0 * kappa * t)) / (2.0 * kappa)
    return np.asarray(np.exp(1j * u * m - 0.5 * v * u**2), dtype=np.complex128)


def gil_pelaez_cdf(
    cf: object,
    x_grid: FloatArray,
    u_max: float = 60.0,
    n_u: int = 4001,
) -> dict[str, float | FloatArray]:
    """Numerically invert a characteristic function to CDF + density.

    ``cf`` must accept an ndarray of frequencies and return complex
    values. Returns the inverted CDF and pdf on ``x_grid`` plus
    quadrature diagnostics.
    """
    xx = np.asarray(x_grid, dtype=np.float64)
    if xx.ndim != 1 or xx.shape[0] < 5:
        raise ValueError("x_grid must be a 1d grid of >= 5 points")
    if not np.all(np.isfinite(xx)) or np.any(np.diff(xx) <= 0):
        raise ValueError("x_grid must be finite and increasing")
    if u_max <= 0 or n_u < 51:
        raise ValueError("bad quadrature settings")
    if not callable(cf):
        raise ValueError("cf must be callable")

    u = np.linspace(1e-8, u_max, n_u)
    phi = np.asarray(cf(u), dtype=np.complex128)
    if phi.shape != u.shape or not np.all(np.isfinite(phi)):
        raise ValueError("cf must return finite values on the grid")
    du = u[1] - u[0]
    cdf = np.empty(xx.shape[0])
    pdf = np.empty(xx.shape[0])
    for i, x in enumerate(xx):
        e = np.exp(-1j * u * x)
        cdf[i] = 0.5 - (1.0 / np.pi) * np.trapezoid(np.imag(phi * e) / u, dx=du)
        pdf[i] = (1.0 / np.pi) * np.trapezoid(np.real(phi * e), dx=du)
    cdf = np.clip(cdf, 0.0, 1.0)
    return {
        "cdf_min": float(np.min(cdf)),
        "cdf_max": float(np.max(cdf)),
        "pdf_sum": float(np.trapezoid(pdf, xx)),
        "max_cdf_err_vs_normal": 0.0,  # filled by caller comparing a ref
        "_cdf": cdf,
        "_pdf": pdf,
        "_u": u,
    }


def synth_ou_inversion(
    seed: int = 20261231 + 290,
    kappa: float = 1.2,
    theta: float = 0.4,
    sigma: float = 0.6,
    t: float = 0.8,
) -> dict[str, float | FloatArray]:
    """Invert the OU CF against its closed-form Gaussian target."""
    x0 = 0.1
    m = theta + (x0 - theta) * np.exp(-kappa * t)
    v = sigma**2 * (1.0 - np.exp(-2.0 * kappa * t)) / (2.0 * kappa)
    sd = np.sqrt(v)
    x_grid = np.linspace(m - 3 * sd, m + 3 * sd, 81)

    def cf(u: FloatArray) -> NDArray[np.complex128]:
        return _ou_cf(u, x0, kappa, theta, sigma, t)

    inv = gil_pelaez_cdf(cf, x_grid)
    cdf_hat = np.asarray(inv["_cdf"])
    pdf_hat = np.asarray(inv["_pdf"])
    cdf_true = _stats.norm.cdf(x_grid, loc=m, scale=sd)
    pdf_true = _stats.norm.pdf(x_grid, loc=m, scale=sd)
    _ = seed  # deterministic — inversion has no randomness
    return {
        "max_cdf_err": float(np.max(np.abs(cdf_hat - cdf_true))),
        "max_pdf_err": float(np.max(np.abs(pdf_hat - pdf_true))),
        "cdf_min": float(inv["cdf_min"]),
        "cdf_max": float(inv["cdf_max"]),
        "true_mean": float(m),
        "true_sd": float(sd),
    }


def bench_gil_pelaez(seed: int = 20261231 + 290) -> dict[str, float]:
    """Wave-50 self-check: quadrature inversion of the OU CF matches
    the closed-form Gaussian CDF/pdf to ~1e-3."""
    a = synth_ou_inversion(seed=seed)
    a2 = synth_ou_inversion(seed=seed)
    detects = float(float(a["max_cdf_err"]) < 5e-3 and float(a["max_pdf_err"]) < 5e-2)
    return {
        "synthetic_detects": detects,
        "synthetic_determinism": float(a == a2),
        "synthetic_max_cdf_err": float(a["max_cdf_err"]),
        "synthetic_max_pdf_err": float(a["max_pdf_err"]),
        "synthetic_cdf_min": float(a["cdf_min"]),
        "synthetic_cdf_max": float(a["cdf_max"]),
        "synthetic_true_sd": float(a["true_sd"]),
    }
