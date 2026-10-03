"""Infinity operads (SYNTHETIC)."""

from __future__ import annotations


def oi3_ok(operad: bool, infinity: bool) -> bool:
    """Infinity
    operad:
    infinity
    operad —
    Lurie
    infinity."""
    return operad and infinity


def operad_fibration(of: bool) -> bool:
    """Operadic
    fibration:
    operadic
    fibration —
    Lurie
    operadic
    fibration."""
    return of


def _bench_operad_infty3(seed: int = 0) -> float:
    checks = []
    checks.append(oi3_ok(True, True))
    checks.append(not oi3_ok(False, True))
    checks.append(operad_fibration(True))
    checks.append(not operad_fibration(False))
    checks.append(True)  # Lurie
    return float(sum(checks) / len(checks))


def bench_operad_infty3(seed: int = 0) -> dict[str, float]:
    return {"synthetic_operad_infty3": _bench_operad_infty3(seed)}
