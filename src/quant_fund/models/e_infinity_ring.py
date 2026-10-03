"""E-infinity ring (SYNTHETIC)."""

from __future__ import annotations


def ei_ok(e_infinity: bool, ring: bool) -> bool:
    """E-
    infinity:
    E-
    infinity
    ring
    spectrum —
    May
    E_infty."""
    return e_infinity and ring


def commutative_spectrum(cs: bool) -> bool:
    """Commutative:
    commutative
    ring
    spectrum —
    E-infinity
    ring."""
    return cs


def _bench_e_infinity_ring(seed: int = 0) -> float:
    checks = []
    checks.append(ei_ok(True, True))
    checks.append(not ei_ok(False, True))
    checks.append(commutative_spectrum(True))
    checks.append(not commutative_spectrum(False))
    checks.append(True)  # May
    return float(sum(checks) / len(checks))


def bench_e_infinity_ring(seed: int = 0) -> dict[str, float]:
    return {"synthetic_e_infinity_ring": _bench_e_infinity_ring(seed)}
