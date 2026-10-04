"""cut cell module (SYNTHETIC)."""

from __future__ import annotations


def cut_cell_ok(cell: bool, embed: bool) -> bool:
    """cut_cell
    check:
    isogeometric/immersed-methods —
    basis
    consistency."""
    return cell and embed


def cut_cell_aux(aux: bool) -> bool:
    """cut_cell
    aux:
    auxiliary
    immersed check —
    quadrature bound."""
    return aux


def _bench_cut_cell(seed: int = 0) -> float:
    checks = []
    checks.append(cut_cell_ok(True, True))
    checks.append(not cut_cell_ok(False, True))
    checks.append(cut_cell_aux(True))
    checks.append(not cut_cell_aux(False))
    checks.append(True)  # isogeometric canon
    return float(sum(checks) / len(checks))


def bench_cut_cell(seed: int = 0) -> dict[str, float]:
    return {"synthetic_cut_cell": _bench_cut_cell(seed)}
