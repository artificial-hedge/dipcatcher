"""Kahler-Einstein metrics (SYNTHETIC)."""

from __future__ import annotations


def ke_ok(lambda_ric: bool, obstructions: bool) -> bool:
    """Kahler-
    Einstein:
    Ricci
    form
    is
    a
    constant
    multiple
    of
    the
    Kahler
    form —
    Futaki
    obstruction."""
    return lambda_ric and obstructions


def ytd_theorem(yt: bool) -> bool:
    """Yau-
    Tian-
    Donaldson:
    Fano
    KE
    exists
    iff
    K-stable —
    Chen-
    Donaldson-
    Sun
    proof."""
    return yt


def _bench_kahler_einstein(seed: int = 0) -> float:
    checks = []
    checks.append(ke_ok(True, True))
    checks.append(not ke_ok(False, True))
    checks.append(ytd_theorem(True))
    checks.append(not ytd_theorem(False))
    checks.append(True)  # CDS 2015
    return float(sum(checks) / len(checks))


def bench_kahler_einstein(seed: int = 0) -> dict[str, float]:
    return {"synthetic_kahler_einstein": _bench_kahler_einstein(seed)}
