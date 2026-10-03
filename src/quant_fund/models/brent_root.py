"""Brent's method: bracketed superlinear root finding (SYNTHETIC)."""

from __future__ import annotations

import numpy as np


def brent(f, a: float, b: float, tol: float = 1e-12, max_iter: int = 100) -> tuple[float, int]:
    fa, fb = f(a), f(b)
    if fa * fb > 0:
        raise ValueError("no bracket")
    c, fc = a, fa
    d = e = b - a
    for it in range(max_iter):
        if fb * fc > 0:
            c, fc = a, fa
            d = e = b - a
        if abs(fc) < abs(fb):
            a, b, c = b, c, b
            fa, fb, fc = fb, fc, fb
        tol1 = 2 * 1e-16 * abs(b) + 0.5 * tol
        xm = 0.5 * (c - b)
        if abs(xm) <= tol1 or fb == 0.0:
            return b, it
        if abs(e) >= tol1 and abs(fa) > abs(fb):
            s = fb / fa
            if a == c:
                p = 2 * xm * s
                q = 1 - s
            else:
                q = fa / fc
                r = fb / fc
                p = s * (2 * xm * q * (q - r) - (b - a) * (r - 1))
                q = (q - 1) * (r - 1) * (s - 1)
            if p > 0:
                q = -q
            p = abs(p)
            if 2 * p < min(3 * xm * q - abs(tol1 * q), abs(e * q)):
                e, d = d, p / q
            else:
                d = e = xm
        else:
            d = e = xm
        a, fa = b, fb
        b += d if abs(d) > tol1 else (tol1 if xm > 0 else -tol1)
        fb = f(b)
    return b, max_iter


def _bench_brent_root(seed: int = 0) -> float:
    checks = []
    # x^3 - x - 2 root ~1.521
    r, it = brent(lambda x: x**3 - x - 2, 1.0, 2.0)
    checks.append(abs(r - 1.5213797068045676) < 1e-10)
    checks.append(it < 15)
    # cos(x) - x = 0 -> Dottie number
    r2, _ = brent(lambda x: np.cos(x) - x, 0.0, 1.0)
    checks.append(abs(r2 - 0.7390851332151607) < 1e-10)
    # flat-ish function: (x-1)^3 -> root 1
    r3, _ = brent(lambda x: (x - 1.0) ** 3, 0.5, 1.5)
    checks.append(abs(r3 - 1.0) < 1e-6)
    # exp(x) - 2: root ln2
    r4, _ = brent(lambda x: np.exp(x) - 2.0, 0.0, 2.0)
    checks.append(abs(r4 - np.log(2.0)) < 1e-10)
    # respects bracket: failure raises
    try:
        brent(lambda x: x**2 + 1, -1.0, 1.0)
        checks.append(False)
    except ValueError:
        checks.append(True)
    return float(sum(checks) / len(checks))


def bench_brent_root(seed: int = 0) -> dict[str, float]:
    return {"synthetic_brent_root": _bench_brent_root(seed)}
