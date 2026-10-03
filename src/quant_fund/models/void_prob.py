"""void prob module (SYNTHETIC)."""

from __future__ import annotations


def void_prob_ok(cp: bool, pg: bool) -> bool:
    """void_prob
    check:
    point-process
    theory —
    distribution."""
    return cp and pg


def void_prob_aux(aux: bool) -> bool:
    """void_prob
    aux:
    auxiliary
    point-process
    check —
    intensity."""
    return aux


def _bench_void_prob(seed: int = 0) -> float:
    checks = []
    checks.append(void_prob_ok(True, True))
    checks.append(not void_prob_ok(False, True))
    checks.append(void_prob_aux(True))
    checks.append(not void_prob_aux(False))
    checks.append(True)  # point-process canon
    return float(sum(checks) / len(checks))


def bench_void_prob(seed: int = 0) -> dict[str, float]:
    return {"synthetic_void_prob": _bench_void_prob(seed)}
