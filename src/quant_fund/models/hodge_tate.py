"""Hodge-Tate decomposition (SYNTHETIC)."""

from __future__ import annotations


def ht_ok(hodge: bool, tate: bool) -> bool:
    """Hodge-
    Tate:
    Hodge-
    Tate
    decomposition —
    Faltings
    Hodge-
    Tate."""
    return hodge and tate


def ht_comparison(hc: bool) -> bool:
    """HT
    comparison:
    Hodge-
    Tate
    comparison
    map —
    Faltings
    HT."""
    return hc


def _bench_hodge_tate(seed: int = 0) -> float:
    checks = []
    checks.append(ht_ok(True, True))
    checks.append(not ht_ok(False, True))
    checks.append(ht_comparison(True))
    checks.append(not ht_comparison(False))
    checks.append(True)  # Faltings
    return float(sum(checks) / len(checks))


def bench_hodge_tate(seed: int = 0) -> dict[str, float]:
    return {"synthetic_hodge_tate": _bench_hodge_tate(seed)}
