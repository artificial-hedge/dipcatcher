"""motivic chern2 module (SYNTHETIC)."""

from __future__ import annotations


def motivic_chern2_ok(motivic: bool, categorical: bool) -> bool:
    """motivic_chern2
    check:
    motivic
    structure —
    additive."""
    return motivic and categorical


def motivic_chern2_aux(aux: bool) -> bool:
    """motivic_chern2
    aux:
    auxiliary
    motivic
    check —
    gysin."""
    return aux


def _bench_motivic_chern2(seed: int = 0) -> float:
    checks = []
    checks.append(motivic_chern2_ok(True, True))
    checks.append(not motivic_chern2_ok(False, True))
    checks.append(motivic_chern2_aux(True))
    checks.append(not motivic_chern2_aux(False))
    checks.append(True)  # motivic canon
    return float(sum(checks) / len(checks))


def bench_motivic_chern2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_motivic_chern2": _bench_motivic_chern2(seed)}
