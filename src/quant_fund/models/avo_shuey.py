"""Shuey three-term AVO approximation + class detection (SYNTHETIC).

R(theta) ~ A + B sin^2(theta) + C (tan^2(theta) - sin^2(theta))
A = intercept (normal incidence), B = gradient, C = curvature term.
"""

from __future__ import annotations

import numpy as np

_SEED = 20261231 + 893


def shuey(theta_deg: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    th = np.radians(np.asarray(theta_deg, dtype=np.float64))
    s2 = np.sin(th) ** 2
    return np.asarray(a + b * s2 + c * (np.tan(th) ** 2 - s2))


def fit_shuey(theta_deg: np.ndarray, r: np.ndarray) -> tuple[float, float, float]:
    th = np.radians(np.asarray(theta_deg, dtype=np.float64))
    s2 = np.sin(th) ** 2
    x = np.column_stack([np.ones_like(s2), s2, np.tan(th) ** 2 - s2])
    coef, *_ = np.linalg.lstsq(x, np.asarray(r, dtype=np.float64), rcond=None)
    return float(coef[0]), float(coef[1]), float(coef[2])


def avo_class(a: float, b: float) -> int:
    """Rutherford-Williams class from intercept/gradient (simplified).

    I: positive intercept, stays positive at 30 deg. IV: positive intercept
    but polarity reverses early (R(30) < 0). III: negative intercept
    (gas sand). II: near-zero intercept.
    """
    if a > 0.03:
        return 4 if a + b * np.sin(np.radians(30.0)) ** 2 < 0 else 1
    if a < -0.03:
        return 3
    return 2


def bench_avo_shuey(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    th = np.linspace(0.0, 40.0, 30)
    score = 0.0
    # recover known Shuey coefficients under noise (class III gas sand)
    a0, b0, c0 = -0.05, -0.18, 0.05
    r = shuey(th, a0, b0, c0) + rng.normal(0, 0.002, th.size)
    a, b, c = fit_shuey(th, r)
    score += 1.0 if abs(a - a0) < 0.01 and abs(b - b0) < 0.03 else 0.0
    score += 1.0 if abs(c - c0) < 0.03 else 0.0
    score += 1.0 if avo_class(a, b) == 3 else 0.0
    # class I: positive intercept, stays positive to 30 deg
    r1 = shuey(th, 0.12, -0.1, 0.0)
    a1, b1, _ = fit_shuey(th, r1)
    score += 1.0 if avo_class(a1, b1) == 1 else 0.0
    # class IV: positive intercept, early polarity reversal
    a4, b4 = 0.06, -0.6
    score += 1.0 if avo_class(a4, b4) == 4 else 0.0
    return {"synthetic_avo_shuey": score / 5.0}
