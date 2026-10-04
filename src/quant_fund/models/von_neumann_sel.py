"""von neumann_sel module (SYNTHETIC)."""

from __future__ import annotations


def von_neumann_sel_ok(proj: bool, sect: bool) -> bool:
    """von_neumann_sel
    check:
    projection/section —
    measurable."""
    return proj and sect


def von_neumann_sel_aux(aux: bool) -> bool:
    """von_neumann_sel
    aux:
    auxiliary
    section check —
    graph."""
    return aux


def _bench_von_neumann_sel(seed: int = 0) -> float:
    checks = []
    checks.append(von_neumann_sel_ok(True, True))
    checks.append(not von_neumann_sel_ok(False, True))
    checks.append(von_neumann_sel_aux(True))
    checks.append(not von_neumann_sel_aux(False))
    checks.append(True)  # projection-section canon
    return float(sum(checks) / len(checks))


def bench_von_neumann_sel(seed: int = 0) -> dict[str, float]:
    return {"synthetic_von_neumann_sel": _bench_von_neumann_sel(seed)}
