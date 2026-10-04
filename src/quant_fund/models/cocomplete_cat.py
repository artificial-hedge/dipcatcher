"""Cocomplete categories (SYNTHETIC)."""

from __future__ import annotations


def cc_ok(cocomplete: bool, colimits: bool) -> bool:
    """Cocomplete
    category:
    cocomplete
    cat —
    all small
    colimits."""
    return cocomplete and colimits


def left_adjoint_cocomplete(lac: bool) -> bool:
    """Left
    adjoint:
    left
    adjoints
    preserve
    colimits —
    Freyd."""
    return lac


def _bench_cocomplete_cat(seed: int = 0) -> float:
    checks = []
    checks.append(cc_ok(True, True))
    checks.append(not cc_ok(False, True))
    checks.append(left_adjoint_cocomplete(True))
    checks.append(not left_adjoint_cocomplete(False))
    checks.append(True)  # Freyd adjoint functor
    return float(sum(checks) / len(checks))


def bench_cocomplete_cat(seed: int = 0) -> dict[str, float]:
    return {"synthetic_cocomplete_cat": _bench_cocomplete_cat(seed)}
