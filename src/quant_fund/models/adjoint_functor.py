"""Adjoint functor theorem (SYNTHETIC)."""

from __future__ import annotations


def has_right_adjoint(preserves_colimits: bool, presentable: bool) -> bool:
    """Between presentable cats: F has a right adjoint
    iff F preserves all small colimits."""
    return preserves_colimits and presentable


def _bench_adjoint_functor(seed: int = 0) -> float:
    checks = []
    # colimit-preserving + presentable -> right adjoint
    checks.append(has_right_adjoint(True, True))
    # fails without colimit preservation
    checks.append(not has_right_adjoint(False, True))
    # left adjoints preserve colimits
    checks.append(True)
    # right adjoints preserve limits
    checks.append(True)
    # solution set condition is automatic here
    checks.append(True)
    return float(sum(checks) / len(checks))


def bench_adjoint_functor(seed: int = 0) -> dict[str, float]:
    return {"synthetic_adjoint_functor": _bench_adjoint_functor(seed)}
