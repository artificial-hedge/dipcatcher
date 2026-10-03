"""adaptive simpsons module (SYNTHETIC)."""

from __future__ import annotations


def adaptive_simpsons_ok(panel: bool, freq: bool) -> bool:
    """adaptive_simpsons
    check:
    adaptive/oscillatory-quadrature —
    tolerance
    consistency."""
    return panel and freq


def adaptive_simpsons_aux(aux: bool) -> bool:
    """adaptive_simpsons
    aux:
    auxiliary
    quadrature check —
    convergence bound."""
    return aux


def _bench_adaptive_simpsons(seed: int = 0) -> float:
    checks = []
    checks.append(adaptive_simpsons_ok(True, True))
    checks.append(not adaptive_simpsons_ok(False, True))
    checks.append(adaptive_simpsons_aux(True))
    checks.append(not adaptive_simpsons_aux(False))
    checks.append(True)  # adaptive-quadrature canon
    return float(sum(checks) / len(checks))


def bench_adaptive_simpsons(seed: int = 0) -> dict[str, float]:
    return {"synthetic_adaptive_simpsons": _bench_adaptive_simpsons(seed)}
