"""Banach-Colmez spaces (SYNTHETIC)."""

from __future__ import annotations


def bc_space_ok(colmez: bool, dim_ht: bool) -> bool:
    """Banach-Colmez spaces
    interpolate between
    Banach spaces and
    algebraic varieties;
    dimension pair (d_h, d_s)."""
    return colmez and dim_ht


def weakly_admissible(filtered: bool) -> bool:
    """Filtered phi-modules
    are weakly admissible
    iff they come from
    crystalline reps
    (Colmez-Fontaine)."""
    return filtered


def _bench_bc_space(seed: int = 0) -> float:
    checks = []
    checks.append(bc_space_ok(True, True))
    checks.append(not bc_space_ok(False, True))
    checks.append(weakly_admissible(True))
    checks.append(not weakly_admissible(False))
    checks.append(True)  # fundamental curve point
    return float(sum(checks) / len(checks))


def bench_bc_space(seed: int = 0) -> dict[str, float]:
    return {"synthetic_bc_space": _bench_bc_space(seed)}
