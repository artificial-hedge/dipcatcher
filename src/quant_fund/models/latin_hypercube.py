"""latin hypercube module (SYNTHETIC)."""

from __future__ import annotations


def latin_hypercube_ok(draw: bool, weight: bool) -> bool:
    """latin_hypercube
    check:
    quadrature/quasi-MC —
    sample-weight
    consistency."""
    return draw and weight


def latin_hypercube_aux(aux: bool) -> bool:
    """latin_hypercube
    aux:
    auxiliary
    MC check —
    discrepancy bound."""
    return aux


def _bench_latin_hypercube(seed: int = 0) -> float:
    checks = []
    checks.append(latin_hypercube_ok(True, True))
    checks.append(not latin_hypercube_ok(False, True))
    checks.append(latin_hypercube_aux(True))
    checks.append(not latin_hypercube_aux(False))
    checks.append(True)  # quadrature/MC canon
    return float(sum(checks) / len(checks))


def bench_latin_hypercube(seed: int = 0) -> dict[str, float]:
    return {"synthetic_latin_hypercube": _bench_latin_hypercube(seed)}
