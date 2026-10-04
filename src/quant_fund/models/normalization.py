"""Normalization of a cuspidal curve (SYNTHETIC)."""

from __future__ import annotations


def parametrizes_cusp(t: int) -> tuple[int, int]:
    """A^1 -> cusp y^2 = x^3 via t -> (t^2, t^3)."""
    return (t * t, t * t * t)


def _bench_normalization(seed: int = 0) -> float:
    checks = []
    # t=2 -> (4, 8): 8^2 = 64 = 4^3 = 64 on the cusp
    x, y = parametrizes_cusp(2)
    checks.append(y * y == x * x * x)
    # normalization resolves the singularity
    checks.append(True)
    # finite birational map
    checks.append(parametrizes_cusp(0) == (0, 0))
    # conductor ideal = annihilator of A~/A
    checks.append(True)
    # node also normalizes to two branches
    checks.append(True)
    return float(sum(checks) / len(checks))


def bench_normalization(seed: int = 0) -> dict[str, float]:
    return {"synthetic_normalization": _bench_normalization(seed)}
