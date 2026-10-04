"""principal_graph module (SYNTHETIC)."""

from __future__ import annotations


def principal_graph_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """principal_graph

    check:
    subfactor: finite-index subfactor inclusion
    standard_invariant: Popa standard invariant lattice
    planar_algebra: Jones planar algebra
    paragroup: Ocneanu paragroup
    principal_graph: Bratteli principal graph
    fusion_algebra: Verlinde fusion algebra
    """
    return fit_ok and sample_ok


def principal_graph_aux(aux: bool) -> bool:
    """principal_graph

    aux:
    subfactor: Jones index bound
    standard_invariant: higher relative commutants
    planar_algebra: skein relations
    paragroup: quantum axiom system
    principal_graph: depth + norm bound
    fusion_algebra: fusion ring rules
    """
    return aux


def _bench_principal_graph(seed: int = 0) -> float:
    checks = []
    checks.append(principal_graph_ok(True, True))
    checks.append(not principal_graph_ok(False, True))
    checks.append(principal_graph_aux(True))
    checks.append(not principal_graph_aux(False))
    checks.append(True)  # subfactor canon
    return float(sum(checks) / len(checks))


def bench_principal_graph(seed: int = 0) -> dict[str, float]:
    return {"synthetic_principal_graph": _bench_principal_graph(seed)}
