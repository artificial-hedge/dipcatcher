"""Quadrature canon: Gauss-Legendre nodes/weights (Newton on P_n), (SYNTHETIC)
Gauss-Hermite, Clenshaw-Curtis (DCT-weight form), and adaptive Simpson.
``bench_quadrature`` gates exactness on known integrals — polynomials
through degree 2n-1, exp, and a smooth bump — versus the analytic answers.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
from numpy.polynomial.legendre import leggauss
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def gauss_legendre(a: float, b: float, n: int) -> tuple[FloatArray, FloatArray]:
    if n < 1 or not a < b:
        raise ValueError("bad interval or order")
    x, w = leggauss(n)
    return 0.5 * (b - a) * x + 0.5 * (a + b), 0.5 * (b - a) * w


def gauss_hermite(n: int) -> tuple[FloatArray, FloatArray]:
    """Nodes/weights for exp(-x^2) integrals on R."""
    if n < 1:
        raise ValueError("n must be positive")
    i = np.arange(1, n)
    off = np.sqrt(i / 2.0)
    j = np.diag(off, 1) + np.diag(off, -1)
    vals, vecs = np.linalg.eigh(j)
    w = np.sqrt(np.pi) * vecs[0, :] ** 2
    return vals.astype(np.float64), w.astype(np.float64)


def clenshaw_curtis(a: float, b: float, n: int) -> tuple[FloatArray, FloatArray]:
    """Clenshaw-Curtis on [a,b] with n+1 nodes (n even for exactness)."""
    if n < 2:
        raise ValueError("n must be >= 2")
    if n % 2 == 1:
        n += 1
    k = np.arange(n + 1)
    th = np.pi * k / n
    x = np.cos(th)
    w = np.zeros(n + 1)
    w[0] = w[n] = 1.0 / (n * n - 1.0) if n % 2 == 0 else 1.0 / (n * n)
    for i in range(1, n):
        s = 0.0
        for j in range(1, n // 2 + 1):
            m = 1.0 if j < n / 2 else 0.5
            s += m * np.cos(2 * j * th[i]) / (4 * j * j - 1)
        w[i] = (2.0 / n) * (1.0 - 2.0 * s)
    # nodes from cos are on [-1,1] descending; map to [a,b]
    xm = 0.5 * (b - a) * x + 0.5 * (a + b)
    wm = 0.5 * (b - a) * w
    order = np.argsort(xm)
    return xm[order], wm[order]


def adaptive_simpson(
    f: Callable,
    a: float,
    b: float,
    tol: float = 1e-10,
    depth: int = 20,
) -> float:
    """Recursive adaptive Simpson."""
    if not a < b:
        raise ValueError("bad interval")

    def simp(fa: float, fm: float, fb: float, a_: float, b_: float) -> float:
        return (b_ - a_) / 6.0 * (fa + 4.0 * fm + fb)

    def rec(a_: float, b_: float, fa: float, fm: float, fb: float, s: float, lvl: int) -> float:
        m = 0.5 * (a_ + b_)
        lm = 0.5 * (a_ + m)
        rm = 0.5 * (m + b_)
        flm = float(f(lm))
        frm = float(f(rm))
        sl = simp(fa, flm, fm, a_, m)
        sr = simp(fm, frm, fb, m, b_)
        err = sl + sr - s
        if lvl <= 0 or abs(err) < 15.0 * tol:
            return sl + sr + err / 15.0
        return rec(a_, m, fa, flm, fm, sl, lvl - 1) + rec(m, b_, fm, frm, fb, sr, lvl - 1)

    fa = float(f(a))
    fb = float(f(b))
    m = 0.5 * (a + b)
    fm = float(f(m))
    return rec(a, b, fa, fm, fb, simp(fa, fm, fb, a, b), depth)


def integrate(f: Callable, method: str, a: float, b: float, n: int = 64) -> float:
    if method == "gauss":
        x, w = gauss_legendre(a, b, n)
        return float(w @ np.asarray([f(v) for v in x]))
    if method == "cc":
        x, w = clenshaw_curtis(a, b, n)
        return float(w @ np.asarray([f(v) for v in x]))
    if method == "simpson":
        return adaptive_simpson(f, a, b)
    raise ValueError(f"unknown method {method}")


def bench_quadrature(seed: int = 20261231) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    out: dict[str, float] = {}
    # Gauss-Legendre is exact for polynomials up to degree 2n-1.
    coeffs = rng.standard_normal(9)
    x, w = gauss_legendre(-2.0, 3.0, 5)
    num = float(w @ np.polyval(coeffs, x))
    ana = float(np.polyval(np.polyint(coeffs), 3.0) - np.polyval(np.polyint(coeffs), -2.0))
    out["synthetic_gl_poly_err"] = float(abs(num - ana))
    # ∫_0^π sin = 2 via three methods.
    for name in ("gauss", "cc", "simpson"):
        out[f"synthetic_{name}_sin_err"] = float(abs(integrate(np.sin, name, 0.0, np.pi, 32) - 2.0))
    # ∫_R x^4 e^{-x^2} dx = 3/4 * sqrt(pi)
    xh, wh = gauss_hermite(16)
    out["synthetic_gh_moment_err"] = float(abs(float(wh @ (xh**4)) - 0.75 * np.sqrt(np.pi)))
    # Adaptive Simpson on a spiky-but-smooth bump.
    bump = lambda t: float(np.exp(-100.0 * (t - 0.35) ** 2))  # noqa: E731
    got = adaptive_simpson(bump, 0.0, 1.0, tol=1e-9)
    want = float(np.sqrt(np.pi) * 0.5 * 0.1 * (math.erf(6.5) + math.erf(3.5)))
    out["synthetic_simpson_bump_err"] = float(abs(got - want))
    return out
