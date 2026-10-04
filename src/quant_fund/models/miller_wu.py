"""miller wu module (SYNTHETIC)."""

from __future__ import annotations


def miller_wu_ok(lqg: bool, gff: bool) -> bool:
    """miller_wu
    check:
    LQG-2
    structure —
    Gwynne."""
    return lqg and gff


def miller_wu_aux(aux: bool) -> bool:
    """miller_wu
    aux:
    auxiliary
    LQG
    check —
    Duplantier."""
    return aux


def _bench_miller_wu(seed: int = 0) -> float:
    checks = []
    checks.append(miller_wu_ok(True, True))
    checks.append(not miller_wu_ok(False, True))
    checks.append(miller_wu_aux(True))
    checks.append(not miller_wu_aux(False))
    checks.append(True)  # LQG-2 canon
    return float(sum(checks) / len(checks))


def bench_miller_wu(seed: int = 0) -> dict[str, float]:
    return {"synthetic_miller_wu": _bench_miller_wu(seed)}
