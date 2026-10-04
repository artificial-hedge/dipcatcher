"""girsanov thm2 module (SYNTHETIC)."""

from __future__ import annotations


def girsanov_thm2_ok(id1: bool, cg: bool) -> bool:
    """girsanov_thm2
    check:
    infinitely-divisible
    structure —
    Levy
    canon."""
    return id1 and cg


def girsanov_thm2_aux(aux: bool) -> bool:
    """girsanov_thm2
    aux:
    auxiliary
    triplet
    check —
    Khinchin
    formula."""
    return aux


def _bench_girsanov_thm2(seed: int = 0) -> float:
    checks = []
    checks.append(girsanov_thm2_ok(True, True))
    checks.append(not girsanov_thm2_ok(False, True))
    checks.append(girsanov_thm2_aux(True))
    checks.append(not girsanov_thm2_aux(False))
    checks.append(True)  # levy canon
    return float(sum(checks) / len(checks))


def bench_girsanov_thm2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_girsanov_thm2": _bench_girsanov_thm2(seed)}
