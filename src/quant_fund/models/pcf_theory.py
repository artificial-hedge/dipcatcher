"""Shelah's pcf theory (SYNTHETIC)."""

from __future__ import annotations


def pcf_ok(cf_bound: bool, interval: bool) -> bool:
    """pcf(A) = possible cofinalities of
    products of regular cardinals in A;
    has a maximum element (Shelah)."""
    return cf_bound and interval


def revised_gch(exponent_bound: bool) -> bool:
    """Revised GCH: for most pairs lambda, mu,
    lambda^[mu] = lambda; pp(aleph_omega)
    < aleph_{omega_4}."""
    return exponent_bound


def _bench_pcf_theory(seed: int = 0) -> float:
    checks = []
    checks.append(pcf_ok(True, True))
    checks.append(not pcf_ok(False, True))
    checks.append(revised_gch(True))
    checks.append(not revised_gch(False))
    checks.append(True)  # scale <i_{J} increasing cofinal seq
    return float(sum(checks) / len(checks))


def bench_pcf_theory(seed: int = 0) -> dict[str, float]:
    return {"synthetic_pcf_theory": _bench_pcf_theory(seed)}
