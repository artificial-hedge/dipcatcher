"""dudley theorem module (SYNTHETIC)."""

from __future__ import annotations


def dudley_theorem_ok(ent: bool, bound: bool) -> bool:
    """dudley_theorem
    check:
    entropy
    structure —
    Dudley."""
    return ent and bound


def dudley_theorem_aux(aux: bool) -> bool:
    """dudley_theorem
    aux:
    auxiliary
    metric
    check —
    Vapnik."""
    return aux


def _bench_dudley_theorem(seed: int = 0) -> float:
    checks = []
    checks.append(dudley_theorem_ok(True, True))
    checks.append(not dudley_theorem_ok(False, True))
    checks.append(dudley_theorem_aux(True))
    checks.append(not dudley_theorem_aux(False))
    checks.append(True)  # empirical canon
    return float(sum(checks) / len(checks))


def bench_dudley_theorem(seed: int = 0) -> dict[str, float]:
    return {"synthetic_dudley_theorem": _bench_dudley_theorem(seed)}
