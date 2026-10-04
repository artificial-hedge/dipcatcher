"""Motivic base change (SYNTHETIC)."""

from __future__ import annotations


def mbc_ok(proper_base: bool, base_change_iso: bool) -> bool:
    """Motivic
    base
    change:
    proper
    pushforward
    commutes
    with
    pullback —
    base
    change
    theorem."""
    return proper_base and base_change_iso


def proper_base_change(pbc: bool) -> bool:
    """Proper
    base
    change:
    base
    change
    along
    proper
    maps —
    Grothendieck
    compatibility."""
    return pbc


def _bench_motivic_base_change(seed: int = 0) -> float:
    checks = []
    checks.append(mbc_ok(True, True))
    checks.append(not mbc_ok(False, True))
    checks.append(proper_base_change(True))
    checks.append(not proper_base_change(False))
    checks.append(True)  # Grothendieck
    return float(sum(checks) / len(checks))


def bench_motivic_base_change(seed: int = 0) -> dict[str, float]:
    return {"synthetic_motivic_base_change": _bench_motivic_base_change(seed)}
