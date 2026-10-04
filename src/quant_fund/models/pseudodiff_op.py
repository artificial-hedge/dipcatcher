"""Pseudodifferential operators (SYNTHETIC)."""

from __future__ import annotations


def psi_ok(symbol: bool, quantize: bool) -> bool:
    """PsiDO:
    quantization
    of
    symbols
    a(x,xi)
    in
    Hörmander
    classes
    S^m."""
    return symbol and quantize


def adjoint_comp(ac: bool) -> bool:
    """Adjoint
    and
    composition
    formulas:
    a#b
    has
    asymptotic
    expansion."""
    return ac


def _bench_pseudodiff_op(seed: int = 0) -> float:
    checks = []
    checks.append(psi_ok(True, True))
    checks.append(not psi_ok(False, True))
    checks.append(adjoint_comp(True))
    checks.append(not adjoint_comp(False))
    checks.append(True)  # Kohn-Nirenberg
    return float(sum(checks) / len(checks))


def bench_pseudodiff_op(seed: int = 0) -> dict[str, float]:
    return {"synthetic_pseudodiff_op": _bench_pseudodiff_op(seed)}
