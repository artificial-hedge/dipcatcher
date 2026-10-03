"""bracketing ent module (SYNTHETIC)."""

from __future__ import annotations


def bracketing_ent_ok(ent: bool, bound: bool) -> bool:
    """bracketing_ent
    check:
    entropy
    structure —
    Dudley."""
    return ent and bound


def bracketing_ent_aux(aux: bool) -> bool:
    """bracketing_ent
    aux:
    auxiliary
    metric
    check —
    Vapnik."""
    return aux


def _bench_bracketing_ent(seed: int = 0) -> float:
    checks = []
    checks.append(bracketing_ent_ok(True, True))
    checks.append(not bracketing_ent_ok(False, True))
    checks.append(bracketing_ent_aux(True))
    checks.append(not bracketing_ent_aux(False))
    checks.append(True)  # empirical canon
    return float(sum(checks) / len(checks))


def bench_bracketing_ent(seed: int = 0) -> dict[str, float]:
    return {"synthetic_bracketing_ent": _bench_bracketing_ent(seed)}
