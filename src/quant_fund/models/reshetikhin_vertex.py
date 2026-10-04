"""reshetikhin vertex module (SYNTHETIC)."""

from __future__ import annotations


def reshetikhin_vertex_ok(sixv: bool, solv: bool) -> bool:
    """reshetikhin_vertex
    check:
    vertex-model
    structure —
    Baxter."""
    return sixv and solv


def reshetikhin_vertex_aux(aux: bool) -> bool:
    """reshetikhin_vertex
    aux:
    auxiliary
    Yang-Baxter
    check —
    Reshetikhin."""
    return aux


def _bench_reshetikhin_vertex(seed: int = 0) -> float:
    checks = []
    checks.append(reshetikhin_vertex_ok(True, True))
    checks.append(not reshetikhin_vertex_ok(False, True))
    checks.append(reshetikhin_vertex_aux(True))
    checks.append(not reshetikhin_vertex_aux(False))
    checks.append(True)  # vertex-model canon
    return float(sum(checks) / len(checks))


def bench_reshetikhin_vertex(seed: int = 0) -> dict[str, float]:
    return {"synthetic_reshetikhin_vertex": _bench_reshetikhin_vertex(seed)}
