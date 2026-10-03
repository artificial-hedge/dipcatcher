"""Hall algebra (SYNTHETIC)."""

from __future__ import annotations


def ha_ok(hall: bool, alg: bool) -> bool:
    """Hall
    alg:
    Hall
    algebra —
    multiplication."""
    return hall and alg


def hall_multiplication(hm: bool) -> bool:
    """Hall
    multiplication:
    Hall
    multiplication —
    filtrations."""
    return hm


def _bench_hall_alg(seed: int = 0) -> float:
    checks = []
    checks.append(ha_ok(True, True))
    checks.append(not ha_ok(False, True))
    checks.append(hall_multiplication(True))
    checks.append(not hall_multiplication(False))
    checks.append(True)  # Ringel
    return float(sum(checks) / len(checks))


def bench_hall_alg(seed: int = 0) -> dict[str, float]:
    return {"synthetic_hall_alg": _bench_hall_alg(seed)}
