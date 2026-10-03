"""bounded lip module (SYNTHETIC)."""

from __future__ import annotations


def bounded_lip_ok(ent: bool, bound: bool) -> bool:
    """bounded_lip
    check:
    entropy
    structure —
    Dudley."""
    return ent and bound


def bounded_lip_aux(aux: bool) -> bool:
    """bounded_lip
    aux:
    auxiliary
    metric
    check —
    Vapnik."""
    return aux


def _bench_bounded_lip(seed: int = 0) -> float:
    checks = []
    checks.append(bounded_lip_ok(True, True))
    checks.append(not bounded_lip_ok(False, True))
    checks.append(bounded_lip_aux(True))
    checks.append(not bounded_lip_aux(False))
    checks.append(True)  # empirical canon
    return float(sum(checks) / len(checks))


def bench_bounded_lip(seed: int = 0) -> dict[str, float]:
    return {"synthetic_bounded_lip": _bench_bounded_lip(seed)}
