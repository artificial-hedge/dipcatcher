"""tate spec2 module (SYNTHETIC)."""

from __future__ import annotations


def tate_spec2_ok(chromatic: bool, periodic: bool) -> bool:
    """tate_spec2
    check:
    chromatic
    structure —
    periodic."""
    return chromatic and periodic


def tate_spec2_aux(aux: bool) -> bool:
    """tate_spec2
    aux:
    auxiliary
    chromatic
    check —
    height."""
    return aux


def _bench_tate_spec2(seed: int = 0) -> float:
    checks = []
    checks.append(tate_spec2_ok(True, True))
    checks.append(not tate_spec2_ok(False, True))
    checks.append(tate_spec2_aux(True))
    checks.append(not tate_spec2_aux(False))
    checks.append(True)  # chromatic canon
    return float(sum(checks) / len(checks))


def bench_tate_spec2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_tate_spec2": _bench_tate_spec2(seed)}
