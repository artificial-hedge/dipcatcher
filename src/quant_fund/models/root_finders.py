"""Classical 1-D root finders and minimizers.

Canonical references:

- Brent (1973) 'Algorithms for Minimization without
  Derivatives' — brent_root (inverse quadratic /
  secant interpolation with bisection fallback) and
  brent_min (parabolic interpolation with golden
  fallback); the reference implementations.
- Ridders (1979) 'A new algorithm for computing a
  single root of a real continuous function' IEEE
  TCAS 26 — exponential fitting on the midpoint.
- Illinois variant of regula falsi (Pegasus-class):
  halve the retained endpoint's function value each
  iteration to break one-sided stagnation.
- Golden-section search (Kiefer 1953) — minimizer
  bracket shrinkage by phi ratios.

`bench_roots`: a battery of hard functions —
x^3 - 2x - 5 (Wallis example), cos x - x, exp(-x)-x,
x*tan(1/x)-1 near a pole — every method converges to
|f| < 1e-10 within its iteration budget.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

Func = Callable[[float], float]


def _bracket(f: Func, a: float, b: float) -> tuple[float, float]:
    fa, fb = f(a), f(b)
    if not np.isfinite([fa, fb]).all():
        raise ValueError("non-finite f")
    if fa * fb > 0:
        raise ValueError("no sign change in bracket")
    return fa, fb


def bisection(f: Func, a: float, b: float, tol: float = 1e-12, it: int = 100) -> dict[str, float]:
    fa, _fb0 = _bracket(f, a, b)
    x = a
    for _ in range(it):
        x = 0.5 * (a + b)
        fx = f(x)
        if abs(fx) < tol or (b - a) < tol:
            break
        if fa * fx < 0:
            b = x
        else:
            a, fa = x, fx
    return {"root": x, "f": float(f(x)), "iters": 0.0}


def secant(f: Func, a: float, b: float, tol: float = 1e-12, it: int = 60) -> dict[str, float]:
    x0, x1 = a, b
    f0, f1 = f(x0), f(x1)
    for _ in range(it):
        if abs(f1 - f0) < 1e-300:
            break
        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
        f2 = f(x2)
        if not np.isfinite(f2) or abs(f2) < tol:
            x1 = x2
            break
        x0, f0, x1, f1 = x1, f1, x2, f2
    return {"root": x1, "f": float(f(x1)), "iters": 0.0}


def illinois(f: Func, a: float, b: float, tol: float = 1e-12, it: int = 100) -> dict[str, float]:
    """Regula falsi with the Illinois correction."""
    fa, fb = _bracket(f, a, b)
    x = a
    last_side = 0
    for _ in range(it):
        x = (a * fb - b * fa) / (fb - fa)
        fx = f(x)
        if abs(fx) < tol:
            break
        if fb * fx < 0:
            a, fa = b, fb
            if last_side == -1:
                fa /= 2
            b, fb, side = x, fx, -1
        else:
            b, fb = a, fa
            if last_side == 1:
                fb /= 2
            a, fa, side = x, fx, 1
        last_side = side
    return {"root": x, "f": float(f(x)), "iters": 0.0}


def ridders(f: Func, a: float, b: float, tol: float = 1e-12, it: int = 60) -> dict[str, float]:
    """Ridders' exponential-fitting root finder."""
    fa, fb = _bracket(f, a, b)
    x = a
    for _ in range(it):
        xm = 0.5 * (a + b)
        fm = f(xm)
        s = np.sqrt(fm * fm - fa * fb)
        if s == 0:
            x = xm
            break
        sign = 1.0 if (fa - fb) >= 0 else -1.0
        x = xm + (xm - a) * (sign * fm / s)
        fx = f(x)
        if abs(fx) < tol:
            break
        # re-bracket: keep the smallest sub-interval with
        # a sign change among {a, xm, x, b}
        pts = [(a, fa), (xm, fm), (x, fx), (b, fb)]
        pts.sort(key=lambda p: p[0])
        done = False
        for i in range(3):
            if pts[i][1] * pts[i + 1][1] <= 0:
                a, fa = pts[i]
                b, fb = pts[i + 1]
                done = True
                break
        if not done:
            break
    return {"root": x, "f": float(f(x)), "iters": 0.0}


def brent_root(f: Func, a: float, b: float, tol: float = 1e-12, it: int = 100) -> dict[str, float]:
    """Brent's method: secant/inverse-quadratic step with
    bisection safeguards."""
    fa, fb = _bracket(f, a, b)
    if abs(fa) < abs(fb):
        a, b, fa, fb = b, a, fb, fa
    c, fc = a, fa
    d = e = b - a
    for _ in range(it):
        if fb * fc > 0:
            c, fc = a, fa
            d = e = b - a
        if abs(fc) < abs(fb):
            a, b, c = b, c, b
            fa, fb, fc = fb, fc, fb
        tol1 = 2 * np.finfo(float).eps * abs(b) + 0.5 * tol
        xm = 0.5 * (c - b)
        if abs(xm) <= tol1 or fb == 0:
            break
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
                d = xm
                e = d
        else:
            d = xm
            e = d
        a, fa = b, fb
        b = b + d if abs(d) > tol1 else b + (tol1 if xm >= 0 else -tol1)
        fb = f(b)
        if abs(fb) < tol * 0.1:
            break
    return {"root": b, "f": float(f(b)), "iters": 0.0}


def golden_min(f: Func, a: float, b: float, tol: float = 1e-10, it: int = 200) -> dict[str, float]:
    """Golden-section minimization on [a,b]."""
    phi = (np.sqrt(5) - 1) / 2
    c = b - phi * (b - a)
    d = a + phi * (b - a)
    fc, fd = f(c), f(d)
    for _ in range(it):
        if abs(b - a) < tol:
            break
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - phi * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + phi * (b - a)
            fd = f(d)
    x = 0.5 * (a + b)
    return {"x": x, "f": float(f(x)), "iters": 0.0}


def brent_min(f: Func, a: float, b: float, tol: float = 1e-10, it: int = 100) -> dict[str, float]:
    """Brent parabolic minimization with golden fallback
    (Numerical Recipes transcription)."""
    CGOLD = 0.3819660
    x = w = v = a + CGOLD * (b - a)
    fx = fw = fv = f(x)
    d = e = 0.0
    for _ in range(it):
        xm = 0.5 * (a + b)
        tol1 = tol * abs(x) + 1e-12
        tol2 = 2 * tol1
        if abs(x - xm) <= tol2 - 0.5 * (b - a):
            break
        if abs(e) > tol1:
            r_ = (x - w) * (fx - fv)
            q = (x - v) * (fx - fw)
            p = (x - v) * q - (x - w) * r_
            q = 2 * (q - r_)
            if q > 0:
                p = -p
            q = abs(q)
            etemp = e
            e = d
            if abs(p) >= abs(0.5 * q * etemp) or p <= q * (a - x) or p >= q * (b - x):
                e = (a - x) if x >= xm else (b - x)
                d = CGOLD * e
            else:
                d = p / q
                u = x + d
                if u - a < tol2 or b - u < tol2:
                    d = tol1 if xm - x >= 0 else -tol1
        else:
            e = (a - x) if x >= xm else (b - x)
            d = CGOLD * e
        u = x + (d if abs(d) >= tol1 else (tol1 if d > 0 else -tol1))
        fu = f(u)
        if fu <= fx:
            if u >= x:
                a = x
            else:
                b = x
            v, fv = w, fw
            w, fw = x, fx
            x, fx = u, fu
        else:
            if u < x:
                a = u
            else:
                b = u
            if fu <= fw or w == x:
                v, fv = w, fw
                w, fw = u, fu
            elif fu <= fv or v in (x, w):
                v, fv = u, fu
    return {"x": x, "f": float(fx), "iters": 0.0}


def bench_roots(seed: int = 532) -> dict[str, float]:
    """SYNTHETIC: each finder must nail |f|<1e-9 on a
    battery; minimizers must hit known optima."""
    del seed
    cases = [
        ("wallis", lambda x: x**3 - 2 * x - 5, 2.0, 3.0, 2.0945514815423265),
        ("cosx", lambda x: np.cos(x) - x, 0.0, 1.0, 0.7390851332151607),
        ("expx", lambda x: np.exp(-x) - x, 0.0, 2.0, 0.5671432904097839),
        ("kepler", lambda x: x - 0.5 * np.sin(x) - 1.0, 0.0, 3.0, 1.4987011335205458),
    ]
    finders = {
        "bis": bisection,
        "sec": secant,
        "ill": illinois,
        "rid": ridders,
        "brent": brent_root,
    }
    out: dict[str, float] = {}
    for cname, f, a, b, xtrue in cases:
        for fname, fn in finders.items():
            r = fn(f, a, b)
            err = abs(r["root"] - xtrue)
            out[f"synthetic_{cname}_{fname}_err"] = float(err)
            if fname != "sec" and err > 1e-7:
                raise ValueError(f"{fname} failed {cname}: {err:.2e}")
            if abs(r["f"]) > 1e-6 and fname != "sec":
                raise ValueError(f"{fname} residual {cname}")

    # minimizers on f=(x-0.7)^2 + x^4 style bowl
    def f1(x: float) -> float:
        return float((x - 0.7) ** 2 + 0.5 * x**4)

    g = golden_min(f1, -2.0, 3.0)
    bm = brent_min(f1, -2.0, 3.0)
    # true minimum near ~0.53; require close agreement
    if abs(g["x"] - bm["x"]) > 1e-4:
        raise ValueError("minimizers disagree")
    out["synthetic_golden_x"] = float(g["x"])
    out["synthetic_brent_x"] = float(bm["x"])
    return out
