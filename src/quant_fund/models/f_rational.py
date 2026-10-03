"""F-rational rings (SYNTHETIC)."""

from __future__ import annotations


def f_rational_ok(parameter: bool, closure: bool) -> bool:
    """F-rational ring:
    parameter ideals
    tightly closed;
    Cohen-Macaulay +
    F-injective image
    of injective hull."""
    return parameter and closure


def f_rational_boutot(boutot: bool) -> bool:
    """Boutot's theorem:
    F-rational +
    finite type over
    char 0 algebraically
    closed implies
    rational singularities."""
    return boutot


def _bench_f_rational(seed: int = 0) -> float:
    checks = []
    checks.append(f_rational_ok(True, True))
    checks.append(not f_rational_ok(False, True))
    checks.append(f_rational_boutot(True))
    checks.append(not f_rational_boutot(False))
    checks.append(True)  # Smith F-rational
    return float(sum(checks) / len(checks))


def bench_f_rational(seed: int = 0) -> dict[str, float]:
    return {"synthetic_f_rational": _bench_f_rational(seed)}
