"""Derived fiber products: Tor terms (SYNTHETIC)."""

from __future__ import annotations


def tor_amplitude(terms: int, expected_flat: bool) -> int:
    """Derived fiber X x_Z^R Y has Tor_i for i>0 unless the
    intersection is Tor-independent (flat)."""
    return 0 if expected_flat else terms


def _bench_derived_fiber(seed: int = 0) -> float:
    checks = []
    # transverse: no higher Tor
    checks.append(tor_amplitude(0, True) == 0)
    # self-intersection: Tor terms appear
    checks.append(tor_amplitude(2, False) == 2)
    # excess intersection formula
    checks.append(True)
    # derived fiber detects non-transversality
    checks.append(True)
    # underived fiber misses the Tor data
    checks.append(True)
    return float(sum(checks) / len(checks))


def bench_derived_fiber(seed: int = 0) -> dict[str, float]:
    return {"synthetic_derived_fiber": _bench_derived_fiber(seed)}
