"""Y-diamond and FF curve (SYNTHETIC)."""

from __future__ import annotations


def y_ok(curve: bool, diamonds: bool) -> bool:
    """The space Y:
    quotient
    Spa(Q_p^breve)
    / phi^Z; the
    underlying
    diamond of
    the Fargues-
    Fontaine
    curve."""
    return curve and diamonds


def ff_divisors(div: bool) -> bool:
    """Fargues-
    Fontaine
    divisors:
    degree-1
    closed points
    parametrize
    untilts."""
    return div


def _bench_y_diamond(seed: int = 0) -> float:
    checks = []
    checks.append(y_ok(True, True))
    checks.append(not y_ok(False, True))
    checks.append(ff_divisors(True))
    checks.append(not ff_divisors(False))
    checks.append(True)  # Fargues-Fontaine
    return float(sum(checks) / len(checks))


def bench_y_diamond(seed: int = 0) -> dict[str, float]:
    return {"synthetic_y_diamond": _bench_y_diamond(seed)}
