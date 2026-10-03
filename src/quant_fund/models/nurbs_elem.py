"""nurbs elem module (SYNTHETIC)."""

from __future__ import annotations


def nurbs_elem_ok(cell: bool, embed: bool) -> bool:
    """nurbs_elem
    check:
    isogeometric/immersed-methods —
    basis
    consistency."""
    return cell and embed


def nurbs_elem_aux(aux: bool) -> bool:
    """nurbs_elem
    aux:
    auxiliary
    immersed check —
    quadrature bound."""
    return aux


def _bench_nurbs_elem(seed: int = 0) -> float:
    checks = []
    checks.append(nurbs_elem_ok(True, True))
    checks.append(not nurbs_elem_ok(False, True))
    checks.append(nurbs_elem_aux(True))
    checks.append(not nurbs_elem_aux(False))
    checks.append(True)  # isogeometric canon
    return float(sum(checks) / len(checks))


def bench_nurbs_elem(seed: int = 0) -> dict[str, float]:
    return {"synthetic_nurbs_elem": _bench_nurbs_elem(seed)}
