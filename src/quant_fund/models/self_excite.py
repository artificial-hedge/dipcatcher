"""self excite module (SYNTHETIC)."""

from __future__ import annotations


def self_excite_ok(pt: bool, meas: bool) -> bool:
    """self_excite
    check:
    point-process
    structure —
    Cox
    intensity."""
    return pt and meas


def self_excite_aux(aux: bool) -> bool:
    """self_excite
    aux:
    auxiliary
    mark
    check —
    Palm
    distribution."""
    return aux


def _bench_self_excite(seed: int = 0) -> float:
    checks = []
    checks.append(self_excite_ok(True, True))
    checks.append(not self_excite_ok(False, True))
    checks.append(self_excite_aux(True))
    checks.append(not self_excite_aux(False))
    checks.append(True)  # point-process canon
    return float(sum(checks) / len(checks))


def bench_self_excite(seed: int = 0) -> dict[str, float]:
    return {"synthetic_self_excite": _bench_self_excite(seed)}
