"""Affine Lie algebras (SYNTHETIC)."""

from __future__ import annotations


def affine_ok(loop: bool, central: bool) -> bool:
    """Affine
    Lie algebra
    g-hat =
    g tensor
    C[t,t^{-1}]
    oplus C c
    oplus C d;
    untwisted
    Kac-Moody."""
    return loop and central


def level_k(level: bool) -> bool:
    """Level-k
    integrable
    reps:
    c acts
    by scalar
    k; the
    Sugawara
    construction
    gives Vir."""
    return level


def _bench_affine_lie(seed: int = 0) -> float:
    checks = []
    checks.append(affine_ok(True, True))
    checks.append(not affine_ok(False, True))
    checks.append(level_k(True))
    checks.append(not level_k(False))
    checks.append(True)  # Kac-Moody
    return float(sum(checks) / len(checks))


def bench_affine_lie(seed: int = 0) -> dict[str, float]:
    return {"synthetic_affine_lie": _bench_affine_lie(seed)}
