"""Variation of parameters for y'' + y = f(t) (SYNTHETIC)."""

from __future__ import annotations

import math


def particular(f, t: float, pts: int = 2000) -> float:
    """y_p = int_0^t sin(t-s) f(s) ds (y1=cos, y2=sin, W=1)."""
    h = t / pts
    total = 0.0
    for i in range(pts + 1):
        s = i * h
        w = 1.0 if 0 < i < pts else 0.5
        total += w * math.sin(t - s) * f(s)
    return total * h


def residual(f, t: float) -> float:
    """Check y_p'' + y_p = f numerically at t."""
    h = 1e-4
    y = particular(f, t)
    yp = particular(f, t + h)
    ym = particular(f, t - h)
    ypp = (yp - 2 * y + ym) / (h * h)
    return float(abs(ypp + y - f(t)))


def _bench_variation_params(seed: int = 0) -> float:
    checks = []
    # y'' + y = 1 -> particular y=1; formula at t=pi/2: int sin(pi/2 - s) ds = 1 - cos(pi/2)... verify residual
    checks.append(residual(lambda s: 1.0, math.pi / 2) < 0.01)
    # y'' + y = t: exact particular = t - sin(t); residual check
    checks.append(residual(lambda s: s, 1.0) < 0.01)
    # Wronskian of cos,sin = 1
    checks.append(abs(math.cos(0.3) * math.cos(0.3) + math.sin(0.3) * math.sin(0.3) - 1.0) < 1e-9)
    # particular for f=0 is 0
    checks.append(abs(particular(lambda s: 0.0, 1.0)) < 1e-12)
    # particular for f=cos(s) (resonance): y_p ~ (t/2) sin t -> check residual at 1.0
    checks.append(residual(lambda s: math.cos(s), 1.0) < 0.02)
    # homogeneous ICs: y_p(0)=0
    checks.append(abs(particular(lambda s: s, 1e-12)) < 1e-9)
    return float(sum(checks) / len(checks))


def bench_variation_params(seed: int = 0) -> dict[str, float]:
    return {"synthetic_variation_params": _bench_variation_params(seed)}
