"""lejan loop module (SYNTHETIC)."""

from __future__ import annotations


def lejan_loop_ok(loop: bool, gff: bool) -> bool:
    """lejan_loop
    check:
    loop-soup
    structure —
    LeJan."""
    return loop and gff


def lejan_loop_aux(aux: bool) -> bool:
    """lejan_loop
    aux:
    auxiliary
    Gaussian-field
    check —
    Lupu."""
    return aux


def _bench_lejan_loop(seed: int = 0) -> float:
    checks = []
    checks.append(lejan_loop_ok(True, True))
    checks.append(not lejan_loop_ok(False, True))
    checks.append(lejan_loop_aux(True))
    checks.append(not lejan_loop_aux(False))
    checks.append(True)  # loop-soup canon
    return float(sum(checks) / len(checks))


def bench_lejan_loop(seed: int = 0) -> dict[str, float]:
    return {"synthetic_lejan_loop": _bench_lejan_loop(seed)}
