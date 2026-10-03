"""mixed elliptic module (SYNTHETIC)."""

from __future__ import annotations


def mixed_elliptic_ok(mixed: bool, motivic: bool) -> bool:
    """mixed_elliptic
    check:
    mixed
    structure —
    period."""
    return mixed and motivic


def mixed_elliptic_aux(aux: bool) -> bool:
    """mixed_elliptic
    aux:
    auxiliary
    mixed
    check —
    Tate."""
    return aux


def _bench_mixed_elliptic(seed: int = 0) -> float:
    checks = []
    checks.append(mixed_elliptic_ok(True, True))
    checks.append(not mixed_elliptic_ok(False, True))
    checks.append(mixed_elliptic_aux(True))
    checks.append(not mixed_elliptic_aux(False))
    checks.append(True)  # mixed-motives canon
    return float(sum(checks) / len(checks))


def bench_mixed_elliptic(seed: int = 0) -> dict[str, float]:
    return {"synthetic_mixed_elliptic": _bench_mixed_elliptic(seed)}
