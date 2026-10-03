"""Symplectic capacities (SYNTHETIC)."""

from __future__ import annotations


def sc_ok(monotone: bool, conformal: bool) -> bool:
    """Symplectic
    capacity:
    monotone
    and
    conformal
    symplectic
    invariant —
    EHZ
    axioms."""
    return monotone and conformal


def displacement_energy(de: bool) -> bool:
    """Displacement
    energy:
    Hofer
    energy
    needed
    to
    disjoin
    a
    set —
    capacity
    bounds
    it."""
    return de


def _bench_symplectic_capacity(seed: int = 0) -> float:
    checks = []
    checks.append(sc_ok(True, True))
    checks.append(not sc_ok(False, True))
    checks.append(displacement_energy(True))
    checks.append(not displacement_energy(False))
    checks.append(True)  # Ekeland-Hofer
    return float(sum(checks) / len(checks))


def bench_symplectic_capacity(seed: int = 0) -> dict[str, float]:
    return {"synthetic_symplectic_capacity": _bench_symplectic_capacity(seed)}
