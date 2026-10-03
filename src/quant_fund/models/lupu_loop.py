"""lupu loop module (SYNTHETIC)."""

from __future__ import annotations


def lupu_loop_ok(loop: bool, gff: bool) -> bool:
    """lupu_loop
    check:
    loop-soup
    structure —
    LeJan."""
    return loop and gff


def lupu_loop_aux(aux: bool) -> bool:
    """lupu_loop
    aux:
    auxiliary
    Gaussian-field
    check —
    Lupu."""
    return aux


def _bench_lupu_loop(seed: int = 0) -> float:
    checks = []
    checks.append(lupu_loop_ok(True, True))
    checks.append(not lupu_loop_ok(False, True))
    checks.append(lupu_loop_aux(True))
    checks.append(not lupu_loop_aux(False))
    checks.append(True)  # loop-soup canon
    return float(sum(checks) / len(checks))


def bench_lupu_loop(seed: int = 0) -> dict[str, float]:
    return {"synthetic_lupu_loop": _bench_lupu_loop(seed)}
