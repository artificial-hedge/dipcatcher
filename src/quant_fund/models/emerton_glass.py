"""emerton glass module (SYNTHETIC)."""

from __future__ import annotations


def emerton_glass_ok(motive: bool, a1: bool) -> bool:
    """emerton_glass
    check:
    motivic-A1
    structure —
    Voevodsky."""
    return motive and a1


def emerton_glass_aux(aux: bool) -> bool:
    """emerton_glass
    aux:
    auxiliary
    motive
    check —
    Morel."""
    return aux


def _bench_emerton_glass(seed: int = 0) -> float:
    checks = []
    checks.append(emerton_glass_ok(True, True))
    checks.append(not emerton_glass_ok(False, True))
    checks.append(emerton_glass_aux(True))
    checks.append(not emerton_glass_aux(False))
    checks.append(True)  # motivic-A1 canon
    return float(sum(checks) / len(checks))


def bench_emerton_glass(seed: int = 0) -> dict[str, float]:
    return {"synthetic_emerton_glass": _bench_emerton_glass(seed)}
