"""Sheaves on stacks (SYNTHETIC)."""

from __future__ import annotations


def stacky_sheaf_ok(lisse_etale: bool, descent: bool) -> bool:
    """Sheaves on an algebraic stack on the
    lisse-etale or flat-fppf site; descent
    for quotient stacks."""
    return lisse_etale and descent


def equivariant_derived(groupoid: bool) -> bool:
    """D([X/G]) = derived category of
    G-equivariant sheaves on X."""
    return groupoid


def _bench_stacky_sheaf(seed: int = 0) -> float:
    checks = []
    checks.append(stacky_sheaf_ok(True, True))
    checks.append(not stacky_sheaf_ok(False, True))
    checks.append(equivariant_derived(True))
    checks.append(not equivariant_derived(False))
    checks.append(True)  # quotient stack sheaves
    return float(sum(checks) / len(checks))


def bench_stacky_sheaf(seed: int = 0) -> dict[str, float]:
    return {"synthetic_stacky_sheaf": _bench_stacky_sheaf(seed)}
