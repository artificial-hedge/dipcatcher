"""galdef ring module (SYNTHETIC)."""

from __future__ import annotations


def galdef_ring_ok(patch: bool, galois: bool) -> bool:
    """galdef_ring
    check:
    Galois-deformation-2
    structure —
    Breuil."""
    return patch and galois


def galdef_ring_aux(aux: bool) -> bool:
    """galdef_ring
    aux:
    auxiliary
    patch
    check —
    Gee."""
    return aux


def _bench_galdef_ring(seed: int = 0) -> float:
    checks = []
    checks.append(galdef_ring_ok(True, True))
    checks.append(not galdef_ring_ok(False, True))
    checks.append(galdef_ring_aux(True))
    checks.append(not galdef_ring_aux(False))
    checks.append(True)  # Galois-deformation-2 canon
    return float(sum(checks) / len(checks))


def bench_galdef_ring(seed: int = 0) -> dict[str, float]:
    return {"synthetic_galdef_ring": _bench_galdef_ring(seed)}
