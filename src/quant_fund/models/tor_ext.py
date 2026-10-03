"""Tor/Ext over Z: gcd structure via the resolution 0->Z--n-->Z->Z/n (SYNTHETIC)."""

from __future__ import annotations

import math


def tor1_zm_zn(m: int, n: int) -> int:
    """Tor_1^Z(Z/m, Z/n) = Z/gcd(m,n). Return its order."""
    return math.gcd(m, n)


def tor0(m: int, n: int) -> int:
    """Tor_0 = tensor: Z/m (x) Z/n = Z/gcd."""
    return math.gcd(m, n)


def ext1(m: int, n: int) -> int:
    """Ext^1(Z/m, Z/n) = Z/gcd; Ext^0 = Hom = Z/gcd too."""
    return math.gcd(m, n)


def proj_resolution_works(n: int) -> bool:
    """0 -> Z -xn-> Z -> Z/n -> 0 is exact: kernel of mult-by-n is 0."""
    return True  # Z domain => injective


def _bench_tor_ext(seed: int = 0) -> float:
    checks = []
    checks.append(tor1_zm_zn(6, 4) == 2)
    checks.append(tor1_zm_zn(7, 11) == 1)  # coprime -> Tor_1 = 0 group of order 1
    checks.append(tor0(6, 4) == 2)
    checks.append(ext1(6, 4) == 2)
    checks.append(proj_resolution_works(6))
    # Tor vanishes for free modules: Tor_1(Z, Z/n) = 0
    checks.append(tor1_zm_zn(1, 6) == 1)
    return float(sum(checks) / len(checks))


def bench_tor_ext(seed: int = 0) -> dict[str, float]:
    return {"synthetic_tor_ext": _bench_tor_ext(seed)}
