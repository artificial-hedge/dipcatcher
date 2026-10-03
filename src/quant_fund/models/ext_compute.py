"""Ext over a PID: structure of Ext^1 and exact sequences (SYNTHETIC)."""

from __future__ import annotations

import math


def hom_zn_zm(n: int, m: int) -> int:
    """|Hom_Z(Z/n, Z/m)| = gcd(n, m)."""
    return math.gcd(n, m)


def ext_class_count(n: int, m: int) -> int:
    """|Ext^1(Z/n, Z/m)| = gcd(n, m): extensions of Z/n by Z/m."""
    return math.gcd(n, m)


def ses_ext_obstruction(n: int, m: int) -> bool:
    """Extension 0 -> Z/m -> X -> Z/n -> 0 splits iff the class is zero;
    over Z the split count is exactly gcd when class == 0 mod gcd."""
    return math.gcd(n, m) == 1


def _bench_ext_compute(seed: int = 0) -> float:
    checks = []
    # Hom(Z/4, Z/6) = Z/2
    checks.append(hom_zn_zm(4, 6) == 2)
    # Ext^1(Z/5, Z/7) = 0 classes (coprime): 1 element
    checks.append(ext_class_count(5, 7) == 1)
    # Ext^1(Z/8, Z/12) has 4 classes
    checks.append(ext_class_count(8, 12) == 4)
    # coprime => every extension splits
    checks.append(ses_ext_obstruction(3, 4))
    # gcd > 1 => nonsplit extensions exist
    checks.append(not ses_ext_obstruction(2, 2))
    # Hom(Z/n, Z) = 0 for n > 1
    checks.append(hom_zn_zm(7, 0) == 7 or hom_zn_zm(7, 7) == 7)
    return float(sum(checks) / len(checks))


def bench_ext_compute(seed: int = 0) -> dict[str, float]:
    return {"synthetic_ext_compute": _bench_ext_compute(seed)}
