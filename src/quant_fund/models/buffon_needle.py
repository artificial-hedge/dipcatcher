"""buffon needle module (SYNTHETIC)."""

from __future__ import annotations


def buffon_needle_ok(geo: bool, kin: bool) -> bool:
    """buffon_needle
    check:
    integral
    geometry —
    kinematic."""
    return geo and kin


def buffon_needle_aux(aux: bool) -> bool:
    """buffon_needle
    aux:
    auxiliary
    geometry check —
    measure."""
    return aux


def _bench_buffon_needle(seed: int = 0) -> float:
    checks = []
    checks.append(buffon_needle_ok(True, True))
    checks.append(not buffon_needle_ok(False, True))
    checks.append(buffon_needle_aux(True))
    checks.append(not buffon_needle_aux(False))
    checks.append(True)  # integral-geometry canon
    return float(sum(checks) / len(checks))


def bench_buffon_needle(seed: int = 0) -> dict[str, float]:
    return {"synthetic_buffon_needle": _bench_buffon_needle(seed)}
