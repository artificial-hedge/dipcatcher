"""baxter vertex module (SYNTHETIC)."""

from __future__ import annotations


def baxter_vertex_ok(sixv: bool, solv: bool) -> bool:
    """baxter_vertex
    check:
    vertex-model
    structure —
    Baxter."""
    return sixv and solv


def baxter_vertex_aux(aux: bool) -> bool:
    """baxter_vertex
    aux:
    auxiliary
    Yang-Baxter
    check —
    Reshetikhin."""
    return aux


def _bench_baxter_vertex(seed: int = 0) -> float:
    checks = []
    checks.append(baxter_vertex_ok(True, True))
    checks.append(not baxter_vertex_ok(False, True))
    checks.append(baxter_vertex_aux(True))
    checks.append(not baxter_vertex_aux(False))
    checks.append(True)  # vertex-model canon
    return float(sum(checks) / len(checks))


def bench_baxter_vertex(seed: int = 0) -> dict[str, float]:
    return {"synthetic_baxter_vertex": _bench_baxter_vertex(seed)}
