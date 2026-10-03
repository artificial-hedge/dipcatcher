"""uniform section module (SYNTHETIC)."""

from __future__ import annotations


def uniform_section_ok(proj: bool, sect: bool) -> bool:
    """uniform_section
    check:
    projection/section —
    measurable."""
    return proj and sect


def uniform_section_aux(aux: bool) -> bool:
    """uniform_section
    aux:
    auxiliary
    section check —
    graph."""
    return aux


def _bench_uniform_section(seed: int = 0) -> float:
    checks = []
    checks.append(uniform_section_ok(True, True))
    checks.append(not uniform_section_ok(False, True))
    checks.append(uniform_section_aux(True))
    checks.append(not uniform_section_aux(False))
    checks.append(True)  # projection-section canon
    return float(sum(checks) / len(checks))


def bench_uniform_section(seed: int = 0) -> dict[str, float]:
    return {"synthetic_uniform_section": _bench_uniform_section(seed)}
