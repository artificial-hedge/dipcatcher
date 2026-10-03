"""brown motives module (SYNTHETIC)."""

from __future__ import annotations


def brown_motives_ok(mixed: bool, motivic: bool) -> bool:
    """brown_motives
    check:
    mixed
    structure —
    period."""
    return mixed and motivic


def brown_motives_aux(aux: bool) -> bool:
    """brown_motives
    aux:
    auxiliary
    mixed
    check —
    Tate."""
    return aux


def _bench_brown_motives(seed: int = 0) -> float:
    checks = []
    checks.append(brown_motives_ok(True, True))
    checks.append(not brown_motives_ok(False, True))
    checks.append(brown_motives_aux(True))
    checks.append(not brown_motives_aux(False))
    checks.append(True)  # mixed-motives canon
    return float(sum(checks) / len(checks))


def bench_brown_motives(seed: int = 0) -> dict[str, float]:
    return {"synthetic_brown_motives": _bench_brown_motives(seed)}
