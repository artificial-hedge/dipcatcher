"""Shifted symplectic structures (SYNTHETIC)."""

from __future__ import annotations


def shifted_sympl_ok(n_shift: int, closed: bool) -> bool:
    """An n-shifted symplectic structure is a closed
    2-form of degree n inducing a nondegenerate
    pairing T -> L[n] (PTVV)."""
    return closed and n_shift >= -1


def lagrangian_fibration(lagrangian: bool) -> bool:
    """Lagrangian morphisms carry isotropic +
    coisotropic data; intersections are shifted."""
    return lagrangian


def _bench_shifted_sympl(seed: int = 0) -> float:
    checks = []
    checks.append(shifted_sympl_ok(0, True))
    checks.append(shifted_sympl_ok(1, True))
    checks.append(not shifted_sympl_ok(0, False))
    checks.append(lagrangian_fibration(True))
    checks.append(not lagrangian_fibration(False))
    return float(sum(checks) / len(checks))


def bench_shifted_sympl(seed: int = 0) -> dict[str, float]:
    return {"synthetic_shifted_sympl": _bench_shifted_sympl(seed)}
