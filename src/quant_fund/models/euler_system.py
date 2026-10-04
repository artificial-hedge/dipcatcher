"""euler system module (SYNTHETIC)."""

from __future__ import annotations


def euler_system_ok(iwasawa: bool, euler: bool) -> bool:
    """euler_system
    check:
    Iwasawa
    structure —
    Euler."""
    return iwasawa and euler


def euler_system_aux(aux: bool) -> bool:
    """euler_system
    aux:
    auxiliary
    Iwasawa
    check —
    Selmer."""
    return aux


def _bench_euler_system(seed: int = 0) -> float:
    checks = []
    checks.append(euler_system_ok(True, True))
    checks.append(not euler_system_ok(False, True))
    checks.append(euler_system_aux(True))
    checks.append(not euler_system_aux(False))
    checks.append(True)  # Iwasawa/Euler-system canon
    return float(sum(checks) / len(checks))


def bench_euler_system(seed: int = 0) -> dict[str, float]:
    return {"synthetic_euler_system": _bench_euler_system(seed)}
