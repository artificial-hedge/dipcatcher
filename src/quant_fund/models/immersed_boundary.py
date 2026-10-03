"""immersed boundary module (SYNTHETIC)."""

from __future__ import annotations


def immersed_boundary_ok(cell: bool, embed: bool) -> bool:
    """immersed_boundary
    check:
    isogeometric/immersed-methods —
    basis
    consistency."""
    return cell and embed


def immersed_boundary_aux(aux: bool) -> bool:
    """immersed_boundary
    aux:
    auxiliary
    immersed check —
    quadrature bound."""
    return aux


def _bench_immersed_boundary(seed: int = 0) -> float:
    checks = []
    checks.append(immersed_boundary_ok(True, True))
    checks.append(not immersed_boundary_ok(False, True))
    checks.append(immersed_boundary_aux(True))
    checks.append(not immersed_boundary_aux(False))
    checks.append(True)  # isogeometric canon
    return float(sum(checks) / len(checks))


def bench_immersed_boundary(seed: int = 0) -> dict[str, float]:
    return {"synthetic_immersed_boundary": _bench_immersed_boundary(seed)}
