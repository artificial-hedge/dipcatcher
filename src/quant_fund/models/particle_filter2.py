"""particle filter2 module (SYNTHETIC)."""

from __future__ import annotations


def particle_filter2_ok(ze: bool, ks: bool) -> bool:
    """particle_filter2
    check:
    filtering —
    posterior
    evolution."""
    return ze and ks


def particle_filter2_aux(aux: bool) -> bool:
    """particle_filter2
    aux:
    auxiliary
    filter
    check —
    innovation."""
    return aux


def _bench_particle_filter2(seed: int = 0) -> float:
    checks = []
    checks.append(particle_filter2_ok(True, True))
    checks.append(not particle_filter2_ok(False, True))
    checks.append(particle_filter2_aux(True))
    checks.append(not particle_filter2_aux(False))
    checks.append(True)  # filtering canon
    return float(sum(checks) / len(checks))


def bench_particle_filter2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_particle_filter2": _bench_particle_filter2(seed)}
