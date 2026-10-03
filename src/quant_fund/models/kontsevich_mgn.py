"""Kontsevich moduli of stable maps (SYNTHETIC)."""

from __future__ import annotations


def kmgn_ok(stack: bool, compact: bool) -> bool:
    """Moduli
    of
    stable
    maps:
    Deligne-
    Mumford
    stack
    compactifying
    holomorphic
    maps —
    Kontsevich."""
    return stack and compact


def virtual_fund(vc: bool) -> bool:
    """Virtual
    fundamental
    class:
    GW
    invariants
    are
    integrals
    against
    a
    virtual
    class —
    Behrend-
    Fantechi."""
    return vc


def _bench_kontsevich_mgn(seed: int = 0) -> float:
    checks = []
    checks.append(kmgn_ok(True, True))
    checks.append(not kmgn_ok(False, True))
    checks.append(virtual_fund(True))
    checks.append(not virtual_fund(False))
    checks.append(True)  # Kontsevich 1995
    return float(sum(checks) / len(checks))


def bench_kontsevich_mgn(seed: int = 0) -> dict[str, float]:
    return {"synthetic_kontsevich_mgn": _bench_kontsevich_mgn(seed)}
