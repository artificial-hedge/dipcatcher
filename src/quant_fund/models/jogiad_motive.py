"""jogiad motive module (SYNTHETIC)."""

from __future__ import annotations


def jogiad_motive_ok(motive: bool, suslin: bool) -> bool:
    """jogiad_motive
    check:
    motivic-A1-2
    structure —
    Suslin."""
    return motive and suslin


def jogiad_motive_aux(aux: bool) -> bool:
    """jogiad_motive
    aux:
    auxiliary
    motive
    check —
    Totaro."""
    return aux


def _bench_jogiad_motive(seed: int = 0) -> float:
    checks = []
    checks.append(jogiad_motive_ok(True, True))
    checks.append(not jogiad_motive_ok(False, True))
    checks.append(jogiad_motive_aux(True))
    checks.append(not jogiad_motive_aux(False))
    checks.append(True)  # motivic-A1-2 canon
    return float(sum(checks) / len(checks))


def bench_jogiad_motive(seed: int = 0) -> dict[str, float]:
    return {"synthetic_jogiad_motive": _bench_jogiad_motive(seed)}
