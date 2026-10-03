"""cross section module (SYNTHETIC)."""

from __future__ import annotations


def cross_section_ok(proj: bool, sect: bool) -> bool:
    """cross_section
    check:
    projection/section —
    measurable."""
    return proj and sect


def cross_section_aux(aux: bool) -> bool:
    """cross_section
    aux:
    auxiliary
    section check —
    graph."""
    return aux


def _bench_cross_section(seed: int = 0) -> float:
    checks = []
    checks.append(cross_section_ok(True, True))
    checks.append(not cross_section_ok(False, True))
    checks.append(cross_section_aux(True))
    checks.append(not cross_section_aux(False))
    checks.append(True)  # projection-section canon
    return float(sum(checks) / len(checks))


def bench_cross_section(seed: int = 0) -> dict[str, float]:
    return {"synthetic_cross_section": _bench_cross_section(seed)}
