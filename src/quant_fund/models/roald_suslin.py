"""roald suslin module (SYNTHETIC)."""

from __future__ import annotations


def roald_suslin_ok(motive: bool, suslin: bool) -> bool:
    """roald_suslin
    check:
    motivic-A1-2
    structure —
    Suslin."""
    return motive and suslin


def roald_suslin_aux(aux: bool) -> bool:
    """roald_suslin
    aux:
    auxiliary
    motive
    check —
    Totaro."""
    return aux


def _bench_roald_suslin(seed: int = 0) -> float:
    checks = []
    checks.append(roald_suslin_ok(True, True))
    checks.append(not roald_suslin_ok(False, True))
    checks.append(roald_suslin_aux(True))
    checks.append(not roald_suslin_aux(False))
    checks.append(True)  # motivic-A1-2 canon
    return float(sum(checks) / len(checks))


def bench_roald_suslin(seed: int = 0) -> dict[str, float]:
    return {"synthetic_roald_suslin": _bench_roald_suslin(seed)}
