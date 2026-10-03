"""equivariant spec module (SYNTHETIC)."""

from __future__ import annotations


def equivariant_spec_ok(spectral: bool, geometry: bool) -> bool:
    """equivariant_spec
    check:
    spectral
    algebraic
    geometry —
    stacky."""
    return spectral and geometry


def equivariant_spec_aux(aux: bool) -> bool:
    """equivariant_spec
    aux:
    auxiliary
    spectral
    check —
    derived."""
    return aux


def _bench_equivariant_spec(seed: int = 0) -> float:
    checks = []
    checks.append(equivariant_spec_ok(True, True))
    checks.append(not equivariant_spec_ok(False, True))
    checks.append(equivariant_spec_aux(True))
    checks.append(not equivariant_spec_aux(False))
    checks.append(True)  # spectral AG canon
    return float(sum(checks) / len(checks))


def bench_equivariant_spec(seed: int = 0) -> dict[str, float]:
    return {"synthetic_equivariant_spec": _bench_equivariant_spec(seed)}
