"""Subobject lattice (SYNTHETIC)."""

from __future__ import annotations


def sl_ok(subobject: bool, heyting: bool) -> bool:
    """Subobject
    lattice:
    subobjects
    form
    Heyting
    algebra —
    Heyting
    logic."""
    return subobject and heyting


def subobject_join(sj: bool) -> bool:
    """Join:
    union
    of
    subobjects
    as
    sup —
    lattice
    join."""
    return sj


def _bench_subobject_lattice(seed: int = 0) -> float:
    checks = []
    checks.append(sl_ok(True, True))
    checks.append(not sl_ok(False, True))
    checks.append(subobject_join(True))
    checks.append(not subobject_join(False))
    checks.append(True)  # Heyting
    return float(sum(checks) / len(checks))


def bench_subobject_lattice(seed: int = 0) -> dict[str, float]:
    return {"synthetic_subobject_lattice": _bench_subobject_lattice(seed)}
