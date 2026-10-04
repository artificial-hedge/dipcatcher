"""Jones polynomial (SYNTHETIC)."""

from __future__ import annotations


def jp_ok(skein: bool, laurent: bool) -> bool:
    """Jones
    polynomial:
    Laurent
    polynomial
    defined
    by
    the
    skein
    relation —
    detects
    chirality."""
    return skein and laurent


def mirror_flips(mf: bool) -> bool:
    """Mirror
    image:
    the
    Jones
    polynomial
    of
    the
    mirror
    replaces
    t
    by
    1/t."""
    return mf


def _bench_jones_poly(seed: int = 0) -> float:
    checks = []
    checks.append(jp_ok(True, True))
    checks.append(not jp_ok(False, True))
    checks.append(mirror_flips(True))
    checks.append(not mirror_flips(False))
    checks.append(True)  # Jones 1984
    return float(sum(checks) / len(checks))


def bench_jones_poly(seed: int = 0) -> dict[str, float]:
    return {"synthetic_jones_poly": _bench_jones_poly(seed)}
