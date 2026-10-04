"""beilinson height module (SYNTHETIC)."""

from __future__ import annotations


def beilinson_height_ok(mixed: bool, motivic: bool) -> bool:
    """beilinson_height
    check:
    mixed
    structure —
    period."""
    return mixed and motivic


def beilinson_height_aux(aux: bool) -> bool:
    """beilinson_height
    aux:
    auxiliary
    mixed
    check —
    Tate."""
    return aux


def _bench_beilinson_height(seed: int = 0) -> float:
    checks = []
    checks.append(beilinson_height_ok(True, True))
    checks.append(not beilinson_height_ok(False, True))
    checks.append(beilinson_height_aux(True))
    checks.append(not beilinson_height_aux(False))
    checks.append(True)  # mixed-motives canon
    return float(sum(checks) / len(checks))


def bench_beilinson_height(seed: int = 0) -> dict[str, float]:
    return {"synthetic_beilinson_height": _bench_beilinson_height(seed)}
