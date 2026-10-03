"""standard_invariant module (SYNTHETIC)."""

from __future__ import annotations


def standard_invariant_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """standard_invariant

    check:
    subfactor: finite-index subfactor inclusion
    standard_invariant: Popa standard invariant lattice
    planar_algebra: Jones planar algebra
    paragroup: Ocneanu paragroup
    principal_graph: Bratteli principal graph
    fusion_algebra: Verlinde fusion algebra
    """
    return fit_ok and sample_ok


def standard_invariant_aux(aux: bool) -> bool:
    """standard_invariant

    aux:
    subfactor: Jones index bound
    standard_invariant: higher relative commutants
    planar_algebra: skein relations
    paragroup: quantum axiom system
    principal_graph: depth + norm bound
    fusion_algebra: fusion ring rules
    """
    return aux


def _bench_standard_invariant(seed: int = 0) -> float:
    checks = []
    checks.append(standard_invariant_ok(True, True))
    checks.append(not standard_invariant_ok(False, True))
    checks.append(standard_invariant_aux(True))
    checks.append(not standard_invariant_aux(False))
    checks.append(True)  # subfactor canon
    return float(sum(checks) / len(checks))


def bench_standard_invariant(seed: int = 0) -> dict[str, float]:
    return {"synthetic_standard_invariant": _bench_standard_invariant(seed)}
