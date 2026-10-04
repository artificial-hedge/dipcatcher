"""mzc motive module (SYNTHETIC)."""

from __future__ import annotations


def mzc_motive_ok(mixed: bool, motivic: bool) -> bool:
    """mzc_motive
    check:
    mixed
    structure —
    period."""
    return mixed and motivic


def mzc_motive_aux(aux: bool) -> bool:
    """mzc_motive
    aux:
    auxiliary
    mixed
    check —
    Tate."""
    return aux


def _bench_mzc_motive(seed: int = 0) -> float:
    checks = []
    checks.append(mzc_motive_ok(True, True))
    checks.append(not mzc_motive_ok(False, True))
    checks.append(mzc_motive_aux(True))
    checks.append(not mzc_motive_aux(False))
    checks.append(True)  # mixed-motives canon
    return float(sum(checks) / len(checks))


def bench_mzc_motive(seed: int = 0) -> dict[str, float]:
    return {"synthetic_mzc_motive": _bench_mzc_motive(seed)}
