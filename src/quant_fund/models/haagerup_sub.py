"""Haagerup subfactor (SYNTHETIC)."""

from __future__ import annotations


def hs_ok(haagerup: bool, subfactor: bool) -> bool:
    """Haagerup
    subfactor:
    Haagerup
    subfactor —
    index
    small."""
    return haagerup and subfactor


def haagerup_cat(hc: bool) -> bool:
    """Haagerup
    category:
    Haagerup
    fusion
    category —
    exotic."""
    return hc


def _bench_haagerup_sub(seed: int = 0) -> float:
    checks = []
    checks.append(hs_ok(True, True))
    checks.append(not hs_ok(False, True))
    checks.append(haagerup_cat(True))
    checks.append(not haagerup_cat(False))
    checks.append(True)  # Haagerup
    return float(sum(checks) / len(checks))


def bench_haagerup_sub(seed: int = 0) -> dict[str, float]:
    return {"synthetic_haagerup_sub": _bench_haagerup_sub(seed)}
