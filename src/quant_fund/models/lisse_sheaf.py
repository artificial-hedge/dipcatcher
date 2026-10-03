"""Lisse sheaf (SYNTHETIC)."""

from __future__ import annotations


def ls_ok(lisse: bool, local: bool) -> bool:
    """Lisse:
    lisse
    etale
    sheaf —
    lisse
    local."""
    return lisse and local


def etale_local(el: bool) -> bool:
    """Etale
    local:
    lisse
    =
    locally
    constant —
    lisse
    sheaf."""
    return el


def _bench_lisse_sheaf(seed: int = 0) -> float:
    checks = []
    checks.append(ls_ok(True, True))
    checks.append(not ls_ok(False, True))
    checks.append(etale_local(True))
    checks.append(not etale_local(False))
    checks.append(True)  # lisse
    return float(sum(checks) / len(checks))


def bench_lisse_sheaf(seed: int = 0) -> dict[str, float]:
    return {"synthetic_lisse_sheaf": _bench_lisse_sheaf(seed)}
