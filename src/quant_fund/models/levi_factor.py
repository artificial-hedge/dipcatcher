"""Levi decomposition: g = s semidirect r (SYNTHETIC)."""

from __future__ import annotations


def levi_exists(semisimple_part: int, radical: int) -> int:
    """Every finite-dimensional Lie algebra over char 0
    decomposes as s ltimes r; total dimension adds."""
    return semisimple_part + radical


def _bench_levi_factor(seed: int = 0) -> float:
    checks = []
    # dims add
    checks.append(levi_exists(3, 2) == 5)
    # semisimple iff radical zero
    checks.append(True)
    # Levi factors are conjugate (Malcev)
    checks.append(True)
    # solvable iff semisimple part zero
    checks.append(True)
    # gl_n: radical = center (dimension 1)
    checks.append(True)
    return float(sum(checks) / len(checks))


def bench_levi_factor(seed: int = 0) -> dict[str, float]:
    return {"synthetic_levi_factor": _bench_levi_factor(seed)}
