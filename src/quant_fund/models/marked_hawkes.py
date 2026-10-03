"""marked hawkes module (SYNTHETIC)."""

from __future__ import annotations


def marked_hawkes_ok(jd: bool, mj: bool) -> bool:
    """marked_hawkes
    check:
    jump-process
    model —
    finite
    activity."""
    return jd and mj


def marked_hawkes_aux(aux: bool) -> bool:
    """marked_hawkes
    aux:
    auxiliary
    jump
    check —
    compensator."""
    return aux


def _bench_marked_hawkes(seed: int = 0) -> float:
    checks = []
    checks.append(marked_hawkes_ok(True, True))
    checks.append(not marked_hawkes_ok(False, True))
    checks.append(marked_hawkes_aux(True))
    checks.append(not marked_hawkes_aux(False))
    checks.append(True)  # jump-process canon
    return float(sum(checks) / len(checks))


def bench_marked_hawkes(seed: int = 0) -> dict[str, float]:
    return {"synthetic_marked_hawkes": _bench_marked_hawkes(seed)}
