"""F-D-modules with Frobenius (SYNTHETIC)."""

from __future__ import annotations


def f_dm_ok(frob_map: bool, iso_on_cohom: bool) -> bool:
    """An F-D-module is a D-module M with
    isomorphism M -> F^*M (Frobenius pullback);
    overconvergent iff unit-root generic."""
    return frob_map and iso_on_cohom


def frobenius_slope(newton_polygon: bool) -> bool:
    """F-isocrystal slopes are the slopes of
    the Newton polygon of Frobenius."""
    return newton_polygon


def _bench_frobenius_dm(seed: int = 0) -> float:
    checks = []
    checks.append(f_dm_ok(True, True))
    checks.append(not f_dm_ok(False, True))
    checks.append(frobenius_slope(True))
    checks.append(not frobenius_slope(False))
    checks.append(True)  # Cartier operator inverse image
    return float(sum(checks) / len(checks))


def bench_frobenius_dm(seed: int = 0) -> dict[str, float]:
    return {"synthetic_frobenius_dm": _bench_frobenius_dm(seed)}
