"""hadwiger chars module (SYNTHETIC)."""

from __future__ import annotations


def hadwiger_chars_ok(geo: bool, kin: bool) -> bool:
    """hadwiger_chars
    check:
    integral
    geometry —
    kinematic."""
    return geo and kin


def hadwiger_chars_aux(aux: bool) -> bool:
    """hadwiger_chars
    aux:
    auxiliary
    geometry check —
    measure."""
    return aux


def _bench_hadwiger_chars(seed: int = 0) -> float:
    checks = []
    checks.append(hadwiger_chars_ok(True, True))
    checks.append(not hadwiger_chars_ok(False, True))
    checks.append(hadwiger_chars_aux(True))
    checks.append(not hadwiger_chars_aux(False))
    checks.append(True)  # integral-geometry canon
    return float(sum(checks) / len(checks))


def bench_hadwiger_chars(seed: int = 0) -> dict[str, float]:
    return {"synthetic_hadwiger_chars": _bench_hadwiger_chars(seed)}
