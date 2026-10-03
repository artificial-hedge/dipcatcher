"""Brouwer degree of self-maps of S^1 (SYNTHETIC)."""

from __future__ import annotations


def deg_power_map(n: int) -> int:
    """deg(z -> z^n) = n on S^1."""
    return n


def deg_compose(f: int, g: int) -> int:
    """deg(f . g) = deg f * deg g."""
    return f * g


def deg_antipodal(d: int) -> int:
    """Antipodal map on S^d has degree (-1)^(d+1)."""
    return 1 if (d + 1) % 2 == 0 else -1


def _bench_mapping_degree(seed: int = 0) -> float:
    checks = []
    checks.append(deg_power_map(3) == 3)
    checks.append(deg_power_map(-1) == -1)
    # deg composition multiplicative
    checks.append(deg_compose(2, 3) == 6)
    # antipodal on S^1 is rotation by pi: degree +1
    checks.append(deg_antipodal(1) == 1)
    # antipodal on S^2: degree -1
    checks.append(deg_antipodal(2) == -1)
    # constant map degree 0
    checks.append(deg_power_map(0) == 0)
    return float(sum(checks) / len(checks))


def bench_mapping_degree(seed: int = 0) -> dict[str, float]:
    return {"synthetic_mapping_degree": _bench_mapping_degree(seed)}
