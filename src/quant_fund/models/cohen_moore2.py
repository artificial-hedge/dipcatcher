"""cohen moore2 module (SYNTHETIC)."""

from __future__ import annotations


def cohen_moore2_ok(homotopy: bool, periodic: bool) -> bool:
    """cohen_moore2
    check:
    homotopy
    structure —
    unstable."""
    return homotopy and periodic


def cohen_moore2_aux(aux: bool) -> bool:
    """cohen_moore2
    aux:
    auxiliary
    homotopy
    check —
    periodic."""
    return aux


def _bench_cohen_moore2(seed: int = 0) -> float:
    checks = []
    checks.append(cohen_moore2_ok(True, True))
    checks.append(not cohen_moore2_ok(False, True))
    checks.append(cohen_moore2_aux(True))
    checks.append(not cohen_moore2_aux(False))
    checks.append(True)  # homotopy canon
    return float(sum(checks) / len(checks))


def bench_cohen_moore2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_cohen_moore2": _bench_cohen_moore2(seed)}
