"""Koszul complex homology for a single element (SYNTHETIC)."""

from __future__ import annotations

import math


def koszul_h0(r_mod: int, x: int) -> int:
    """For R = Z/r and element x, H_0 K(x) = R/xR; |xR| = r/gcd(r, x),
    so the quotient has size gcd(r, x)."""
    return math.gcd(r_mod, x)


def koszul_h1(r_mod: int, x: int) -> int:
    """H_1 K(x; R) = Ann_R(x) has size gcd(r, x) for R = Z/r."""
    return math.gcd(r_mod, x)


def _bench_koszul_homology(seed: int = 0) -> float:
    checks = []
    # x = 2 in Z/12: xR = even residues size 6, H_0 = Z/2 (size 2)
    checks.append(koszul_h0(12, 2) == 2)
    # Ann(2) in Z/12 = multiples of 6: size 6
    checks.append(koszul_h1(12, 2) == 2)
    # unit element: H_0 = 0 (size 1), H_1 = 0
    checks.append(koszul_h0(12, 5) == 1)
    checks.append(koszul_h1(12, 5) == 1)
    # x = 0: H_0 = R size r, H_1 = R size r
    checks.append(koszul_h0(12, 0) == 12)
    checks.append(koszul_h1(12, 0) == 12)
    return float(sum(checks) / len(checks))


def bench_koszul_homology(seed: int = 0) -> dict[str, float]:
    return {"synthetic_koszul_homology": _bench_koszul_homology(seed)}
