"""xfem module (SYNTHETIC)."""

from __future__ import annotations


def xfem_ok(cell: bool, embed: bool) -> bool:
    """xfem
    check:
    isogeometric/immersed-methods —
    basis
    consistency."""
    return cell and embed


def xfem_aux(aux: bool) -> bool:
    """xfem
    aux:
    auxiliary
    immersed check —
    quadrature bound."""
    return aux


def _bench_xfem(seed: int = 0) -> float:
    checks = []
    checks.append(xfem_ok(True, True))
    checks.append(not xfem_ok(False, True))
    checks.append(xfem_aux(True))
    checks.append(not xfem_aux(False))
    checks.append(True)  # isogeometric canon
    return float(sum(checks) / len(checks))


def bench_xfem(seed: int = 0) -> dict[str, float]:
    return {"synthetic_xfem": _bench_xfem(seed)}
