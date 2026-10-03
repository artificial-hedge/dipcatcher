"""Tor over Z and flatness of Q (SYNTHETIC)."""

from __future__ import annotations

import math


def tensor_zn_zm(n: int, m: int) -> int:
    """Z/n (x) Z/m = Z/gcd(n, m) (order; Z/0 = Z gives n)."""
    if n == 0:
        return m
    if m == 0:
        return n
    return math.gcd(n, m)


def tor1_zn_zm(n: int, m: int) -> int:
    """Tor_1(Z/n, Z/m) = Z/gcd(n, m)."""
    return math.gcd(n, m)


def flat_over_z(torsion_free: bool) -> bool:
    """Over Z a module is flat iff it is torsion-free."""
    return torsion_free


def _bench_tor_compute(seed: int = 0) -> float:
    checks = []
    # Z/4 (x) Z/6 = Z/2
    checks.append(tensor_zn_zm(4, 6) == 2)
    # Z/3 (x) Z/5 = 0 (order 1)
    checks.append(tensor_zn_zm(3, 5) == 1)
    # Tor_1(Z/2, Z/4) = Z/2
    checks.append(tor1_zn_zm(2, 4) == 2)
    # Tor_1 coprime vanishes
    checks.append(tor1_zn_zm(3, 7) == 1)
    # Z (x) Z/n = Z/n
    checks.append(tensor_zn_zm(0, 9) == 9)
    # Q is flat over Z
    checks.append(flat_over_z(True))
    checks.append(not flat_over_z(False))
    return float(sum(checks) / len(checks))


def bench_tor_compute(seed: int = 0) -> dict[str, float]:
    return {"synthetic_tor_compute": _bench_tor_compute(seed)}
