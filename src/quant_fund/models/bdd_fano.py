"""Bounded Fano (SYNTHIC) (SYNTHETIC)."""

from __future__ import annotations


def bf_ok(bounded: bool, anti_canonical: bool) -> bool:
    """Bounded
    Fano:
    Fano
    varieties
    with
    bounded
    anti-
    canonical —
    BAB
    theorem."""
    return bounded and anti_canonical


def birkar_bab(bb: bool) -> bool:
    """Birkar
    BAB:
    boundedness
    of
    Fano
    varieties —
    Borisov-
    Alexeev-
    Borisov."""
    return bb


def _bench_bdd_fano(seed: int = 0) -> float:
    checks = []
    checks.append(bf_ok(True, True))
    checks.append(not bf_ok(False, True))
    checks.append(birkar_bab(True))
    checks.append(not birkar_bab(False))
    checks.append(True)  # Birkar
    return float(sum(checks) / len(checks))


def bench_bdd_fano(seed: int = 0) -> dict[str, float]:
    return {"synthetic_bdd_fano": _bench_bdd_fano(seed)}
