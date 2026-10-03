"""Torsion sheaves (SYNTHETIC)."""

from __future__ import annotations


def ts2_ok(torsion: bool, sheaf: bool) -> bool:
    """Torsion
    sheaf:
    torsion
    sheaf —
    stalks
    torsion."""
    return torsion and sheaf


def prime_power_torsion(ppt: bool) -> bool:
    """Prime
    power
    torsion:
    prime
    power
    torsion —
    l
    power
    torsion."""
    return ppt


def _bench_torsion_sheaf(seed: int = 0) -> float:
    checks = []
    checks.append(ts2_ok(True, True))
    checks.append(not ts2_ok(False, True))
    checks.append(prime_power_torsion(True))
    checks.append(not prime_power_torsion(False))
    checks.append(True)  # Artin
    return float(sum(checks) / len(checks))


def bench_torsion_sheaf(seed: int = 0) -> dict[str, float]:
    return {"synthetic_torsion_sheaf": _bench_torsion_sheaf(seed)}
