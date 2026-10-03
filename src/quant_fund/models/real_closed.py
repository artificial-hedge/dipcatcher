"""Real-closed field axioms (SYNTHETIC)."""

from __future__ import annotations


def sign_change_root(coeffs: list[float]) -> bool:
    """Odd-degree real polynomial has a real root (intermediate value)."""
    deg = len(coeffs) - 1
    if deg % 2 == 0:
        return False
    lo, hi = -100.0, 100.0

    def p(x: float) -> float:
        return float(sum(c * x**i for i, c in enumerate(coeffs)))

    for _ in range(200):
        mid = (lo + hi) / 2
        if p(lo) * p(mid) <= 0:
            hi = mid
        else:
            lo = mid
    return abs(p((lo + hi) / 2)) < 1e-6


def _bench_real_closed(seed: int = 0) -> float:
    checks = []
    # x^3 - 2 = 0 has real root cube-root of 2
    checks.append(sign_change_root([-2.0, 0.0, 0.0, 1.0]))
    # x^3 - x has roots
    checks.append(sign_change_root([0.0, -1.0, 0.0, 1.0]))
    # x - 1
    checks.append(sign_change_root([-1.0, 1.0]))
    # squares are a positive cone in R: x^2 >= 0
    checks.append(all(x * x >= 0 for x in [-3.0, -0.5, 0.0, 2.0]))
    # every positive element is a square (in R): sqrt exists
    checks.append(abs(4.0**0.5 - 2.0) < 1e-9)
    return float(sum(checks) / len(checks))


def bench_real_closed(seed: int = 0) -> dict[str, float]:
    return {"synthetic_real_closed": _bench_real_closed(seed)}
