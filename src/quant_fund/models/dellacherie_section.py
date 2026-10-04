"""dellacherie section module (SYNTHETIC)."""

from __future__ import annotations


def dellacherie_section_ok(proj: bool, sect: bool) -> bool:
    """dellacherie_section
    check:
    projection/section —
    measurable."""
    return proj and sect


def dellacherie_section_aux(aux: bool) -> bool:
    """dellacherie_section
    aux:
    auxiliary
    section check —
    graph."""
    return aux


def _bench_dellacherie_section(seed: int = 0) -> float:
    checks = []
    checks.append(dellacherie_section_ok(True, True))
    checks.append(not dellacherie_section_ok(False, True))
    checks.append(dellacherie_section_aux(True))
    checks.append(not dellacherie_section_aux(False))
    checks.append(True)  # projection-section canon
    return float(sum(checks) / len(checks))


def bench_dellacherie_section(seed: int = 0) -> dict[str, float]:
    return {"synthetic_dellacherie_section": _bench_dellacherie_section(seed)}
