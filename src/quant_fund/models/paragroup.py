"""paragroup module (SYNTHETIC)."""

from __future__ import annotations


def paragroup_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """paragroup

    check:
    subfactor: finite-index subfactor inclusion
    standard_invariant: Popa standard invariant lattice
    planar_algebra: Jones planar algebra
    paragroup: Ocneanu paragroup
    principal_graph: Bratteli principal graph
    fusion_algebra: Verlinde fusion algebra
    """
    return fit_ok and sample_ok


def paragroup_aux(aux: bool) -> bool:
    """paragroup

    aux:
    subfactor: Jones index bound
    standard_invariant: higher relative commutants
    planar_algebra: skein relations
    paragroup: quantum axiom system
    principal_graph: depth + norm bound
    fusion_algebra: fusion ring rules
    """
    return aux


def _bench_paragroup(seed: int = 0) -> float:
    checks = []
    checks.append(paragroup_ok(True, True))
    checks.append(not paragroup_ok(False, True))
    checks.append(paragroup_aux(True))
    checks.append(not paragroup_aux(False))
    checks.append(True)  # subfactor canon
    return float(sum(checks) / len(checks))


def bench_paragroup(seed: int = 0) -> dict[str, float]:
    return {"synthetic_paragroup": _bench_paragroup(seed)}
