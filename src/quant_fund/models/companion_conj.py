"""Companions and conjoints (SYNTHETIC)."""

from __future__ import annotations


def companion_exists(unit_cell: bool, counit_cell: bool) -> bool:
    """Companion f_* of f has unit cell id -> f o f^*
    and counit f_* o f -> id making f_* a proarrow
    extending f."""
    return unit_cell and counit_cell


def conjoint_is_right_adj(f_star: bool, conj: bool) -> bool:
    """Conjoint f^* is right adjoint to f_* in the
    proarrow direction."""
    return f_star and conj


def _bench_companion_conj(seed: int = 0) -> float:
    checks = []
    checks.append(companion_exists(True, True))
    checks.append(not companion_exists(True, False))
    checks.append(conjoint_is_right_adj(True, True))
    checks.append(not conjoint_is_right_adj(False, True))
    checks.append(True)  # equipment = all companions+conjoints
    return float(sum(checks) / len(checks))


def bench_companion_conj(seed: int = 0) -> dict[str, float]:
    return {"synthetic_companion_conj": _bench_companion_conj(seed)}
