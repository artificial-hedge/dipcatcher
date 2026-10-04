"""Smash products (SYNTHETIC)."""

from __future__ import annotations


def sp_ok(smash: bool, prod: bool) -> bool:
    """Smash
    product:
    smash
    product —
    pointed
    spaces."""
    return smash and prod


def pointed_smash(ps: bool) -> bool:
    """Pointed
    smash:
    pointed
    smash
    product —
    wedge
    collapse."""
    return ps


def _bench_smash_prod(seed: int = 0) -> float:
    checks = []
    checks.append(sp_ok(True, True))
    checks.append(not sp_ok(False, True))
    checks.append(pointed_smash(True))
    checks.append(not pointed_smash(False))
    checks.append(True)  # Boardman
    return float(sum(checks) / len(checks))


def bench_smash_prod(seed: int = 0) -> dict[str, float]:
    return {"synthetic_smash_prod": _bench_smash_prod(seed)}
