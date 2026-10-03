"""Primitive element theorem: Q(sqrt2, sqrt3) = Q(sqrt2 + sqrt3) (SYNTHETIC)."""

from __future__ import annotations

import numpy as np


def _bench_primitive_elem(seed: int = 0) -> float:
    checks = []
    s2, s3 = np.sqrt(2.0), np.sqrt(3.0)
    a = s2 + s3
    # minimal polynomial of sqrt2+sqrt3: x^4 - 10x^2 + 1
    checks.append(abs(a**4 - 10 * a**2 + 1) < 1e-9)
    # degree 4 is exact: no smaller polynomial (a, a^2, a^3 linearly independent
    # over rationals w.r.t. basis {1, sqrt2, sqrt3, sqrt6})
    v0 = np.array([1.0, 0.0, 0.0, 0.0])
    v1 = np.array([0.0, 1.0, 1.0, 0.0])  # a
    v2 = np.array([5.0, 0.0, 0.0, 2.0])  # a^2 = 5 + 2 sqrt6
    v3 = np.array([0.0, 11.0, 9.0, 0.0])  # a^3 = 11 sqrt2 + 9 sqrt3
    m = np.stack([v0, v1, v2, v3])
    checks.append(abs(float(np.linalg.det(m))) > 1e-9)
    # sqrt2 in Q(a): a^3 - 9a = 2 sqrt2 -> sqrt2 = (a^3 - 9a)/2
    checks.append(abs((a**3 - 9 * a) / 2 - s2) < 1e-9)
    # sqrt3 similarly: (11a - a^3)/2
    checks.append(abs((11 * a - a**3) / 2 - s3) < 1e-9)
    # conjugates +-sqrt2 +-sqrt3 are all roots of x^4 - 10x^2 + 1
    for eps in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
        r = eps[0] * s2 + eps[1] * s3
        checks.append(abs(r**4 - 10 * r**2 + 1) < 1e-9)
    return float(sum(checks) / len(checks))


def bench_primitive_elem(seed: int = 0) -> dict[str, float]:
    return {"synthetic_primitive_elem": _bench_primitive_elem(seed)}
