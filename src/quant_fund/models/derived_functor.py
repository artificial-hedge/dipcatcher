"""Derived functors over Z via the standard 2-term resolution (SYNTHETIC)."""

from __future__ import annotations

import math


def ext_zn_zm(n: int, m: int, i: int) -> int:
    """Order of Ext^i_Z(Z/n, Z/m): Z/gcd for i in {0,1}, else 0."""
    if i in (0, 1):
        return math.gcd(n, m)
    return 1


def tor_zn_zm(n: int, m: int, i: int) -> int:
    """Order of Tor_i^Z(Z/n, Z/m): Z/gcd for i in {0,1}, else 0."""
    if i in (0, 1):
        return math.gcd(n, m)
    return 1


def _bench_derived_functor(seed: int = 0) -> float:
    checks = []
    # Ext^1(Z/2, Z/4) = Z/2
    checks.append(ext_zn_zm(2, 4, 1) == 2)
    # Ext^1(Z/6, Z/9) = Z/3
    checks.append(ext_zn_zm(6, 9, 1) == 3)
    # Ext^i = 0 for i >= 2 over Z (global dimension 1)
    checks.append(ext_zn_zm(4, 4, 2) == 1)
    # Hom(Z/n, Z/m) = Z/gcd
    checks.append(ext_zn_zm(8, 12, 0) == 4)
    # coprime => Ext^1 = 0
    checks.append(ext_zn_zm(3, 5, 1) == 1)
    # Tor_1(Z/2, Z/2) = Z/2; Tor_i = 0 for i >= 2
    checks.append(tor_zn_zm(2, 2, 1) == 2)
    checks.append(tor_zn_zm(4, 6, 3) == 1)
    return float(sum(checks) / len(checks))


def bench_derived_functor(seed: int = 0) -> dict[str, float]:
    return {"synthetic_derived_functor": _bench_derived_functor(seed)}
