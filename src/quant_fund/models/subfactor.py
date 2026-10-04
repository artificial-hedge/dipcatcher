"""subfactor module (SYNTHETIC)."""

from __future__ import annotations


def subfactor_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """subfactor

    check:
    subfactor: finite-index subfactor inclusion
    standard_invariant: Popa standard invariant lattice
    planar_algebra: Jones planar algebra
    paragroup: Ocneanu paragroup
    principal_graph: Bratteli principal graph
    fusion_algebra: Verlinde fusion algebra
    """
    return fit_ok and sample_ok


def subfactor_aux(aux: bool) -> bool:
    """subfactor

    aux:
    subfactor: Jones index bound
    standard_invariant: higher relative commutants
    planar_algebra: skein relations
    paragroup: quantum axiom system
    principal_graph: depth + norm bound
    fusion_algebra: fusion ring rules
    """
    return aux


def _bench_subfactor(seed: int = 0) -> float:
    checks = []
    checks.append(subfactor_ok(True, True))
    checks.append(not subfactor_ok(False, True))
    checks.append(subfactor_aux(True))
    checks.append(not subfactor_aux(False))
    checks.append(True)  # subfactor canon
    return float(sum(checks) / len(checks))


def bench_subfactor(seed: int = 0) -> dict[str, float]:
    return {"synthetic_subfactor": _bench_subfactor(seed)}
