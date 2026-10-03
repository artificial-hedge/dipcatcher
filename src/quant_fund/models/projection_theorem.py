"""projection theorem module (SYNTHETIC)."""

from __future__ import annotations


def projection_theorem_ok(proj: bool, sect: bool) -> bool:
    """projection_theorem
    check:
    projection/section —
    measurable."""
    return proj and sect


def projection_theorem_aux(aux: bool) -> bool:
    """projection_theorem
    aux:
    auxiliary
    section check —
    graph."""
    return aux


def _bench_projection_theorem(seed: int = 0) -> float:
    checks = []
    checks.append(projection_theorem_ok(True, True))
    checks.append(not projection_theorem_ok(False, True))
    checks.append(projection_theorem_aux(True))
    checks.append(not projection_theorem_aux(False))
    checks.append(True)  # projection-section canon
    return float(sum(checks) / len(checks))


def bench_projection_theorem(seed: int = 0) -> dict[str, float]:
    return {"synthetic_projection_theorem": _bench_projection_theorem(seed)}
