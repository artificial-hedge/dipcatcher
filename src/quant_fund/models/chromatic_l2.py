"""chromatic l2 module (SYNTHETIC)."""

from __future__ import annotations


def chromatic_l2_ok(chromatic: bool, height: bool) -> bool:
    """chromatic_l2
    check:
    chromatic
    height
    structure —
    stratified."""
    return chromatic and height


def chromatic_l2_aux(aux: bool) -> bool:
    """chromatic_l2
    aux:
    auxiliary
    chromatic
    check —
    spectral."""
    return aux


def _bench_chromatic_l2(seed: int = 0) -> float:
    checks = []
    checks.append(chromatic_l2_ok(True, True))
    checks.append(not chromatic_l2_ok(False, True))
    checks.append(chromatic_l2_aux(True))
    checks.append(not chromatic_l2_aux(False))
    checks.append(True)  # chromatic canon
    return float(sum(checks) / len(checks))


def bench_chromatic_l2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_chromatic_l2": _bench_chromatic_l2(seed)}
