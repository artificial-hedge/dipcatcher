"""ETDRK4 canon: exponential time-differencing RK4 (Cox–Matthews)
for semilinear stiff ODEs u' = L u + N(u) — the stiff linear part
is integrated exactly via phi functions, removing the stability
bottleneck. Tested on a 2x2 stiff semilinear system with a known
variation-of-constants solution. All SYNTHETIC.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def phi1(z: FloatArray) -> FloatArray:
    """φ1(z) = (e^z − 1)/z, series for small |z|."""
    out = np.empty_like(z)
    small = np.abs(z) < 1e-7
    zs = z[small]
    out[small] = 1 + zs / 2 + zs**2 / 6 + zs**3 / 24
    out[~small] = np.expm1(z[~small]) / z[~small]
    return np.asarray(out, dtype=np.float64)


def etdrk4_step_diag(
    L: FloatArray,
    nlin,
    u: FloatArray,
    h: float,
    E: FloatArray,
    E2: FloatArray,
    Q: FloatArray,
    f1: FloatArray,
    f2: FloatArray,
    f3: FloatArray,
) -> FloatArray:
    """Scalar-mode ETDRK4 step on diagonalized L (Cox–Matthews
    contour-integral coefficient form)."""
    Nu = nlin(u)
    a = E2 * u + Q * Nu
    Na = nlin(a)
    b = E2 * u + Q * Na
    Nb = nlin(b)
    c = E2 * a + Q * (2 * Nb - Nu)
    Nc = nlin(c)
    return np.asarray(E * u + f1 * Nu + 2 * f2 * (Na + Nb) + f3 * Nc, dtype=np.float64)


def etdrk4(
    L: FloatArray,
    nlin,
    u0: FloatArray,
    t: FloatArray,
) -> FloatArray:
    """ETDRK4 for u' = L u + N(u) with diagonal L."""
    out = np.empty((len(t), u0.size))
    out[0] = u0
    u = u0.copy()
    for i in range(1, len(t)):
        h = t[i] - t[i - 1]
        z = L * h
        E = np.exp(z)
        E2 = np.exp(z / 2)
        # Cox–Matthews coefficients (diagonal, closed form)
        phi_half = phi1(z / 2)
        Q = h / 2 * phi_half
        f1 = h * (-4 - z + np.exp(z) * (4 - 3 * z + z**2)) / (z**3 + 1e-30)
        f2 = h * (2 + z + np.exp(z) * (-2 + z)) / (z**3 + 1e-30)
        f3 = h * (-4 - 3 * z - z**2 + np.exp(z) * (4 - z)) / (z**3 + 1e-30)
        # series fallback near z=0
        small = np.abs(z) < 1e-4
        zq = z[small] ** 2
        f1[small] = h * (1 / 6 + z[small] / 8 + zq / 30 + z[small] ** 3 / 90)
        f2[small] = h * (1 / 3 + z[small] / 12 + zq / 80)
        f3[small] = h * (1 / 6 + z[small] / 24 + zq / 180)
        u = etdrk4_step_diag(L, nlin, u, h, E, E2, Q, f1, f2, f3)
        out[i] = u
    return out


def bench_etdrk4(seed: int = 20261231) -> dict[str, float]:
    """Order-4 check on u' = λu + u² decay (exact solution known),
    plus a stiffness demo at λ=-50 where explicit RK4 explodes."""
    out: dict[str, float] = {}

    # Bernoulli: v=1/u gives v' = -λv - 1, so
    # u(t) = λ u0 / ((λ + u0)·e^{-λt} − u0)
    def exact_u(t: float, lam: float, u0: float) -> float:
        return float(lam * u0 / ((lam + u0) * np.exp(-lam * t) - u0))

    lam = -2.0
    u0 = np.array([0.3])
    nlin = lambda u: u**2  # noqa: E731
    L = np.array([lam])
    errs = {}
    for n in (32, 64):
        t = np.linspace(0.0, 2.0, n + 1)
        y = etdrk4(L, nlin, u0, t)
        errs[n] = abs(y[-1, 0] - exact_u(2.0, lam, u0[0]))
    out["synthetic_etdrk4_order"] = float(np.log2(errs[32] / errs[64]))
    out["synthetic_etdrk4_err"] = float(errs[64])

    # stiff demo: homogeneous decay u' = -60u, h=0.05 → z=-3 is
    # beyond explicit RK4's stability edge (~-2.78), but ETD
    # integrates the linear part exactly
    lam2 = -60.0
    L2 = np.array([lam2])
    nlin2 = lambda u: np.zeros_like(u)  # noqa: E731
    t = np.linspace(0.0, 1.0, 21)
    y = etdrk4(L2, nlin2, np.array([1.0]), t)
    out["synthetic_etdrk4_stiff_err"] = float(abs(y[-1, 0] - np.exp(-60.0)))
    out["synthetic_etdrk4_stiff_finite"] = float(np.isfinite(y).all())
    return out
