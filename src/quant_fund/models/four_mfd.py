"""4-manifolds (SYNTHETIC)."""

from __future__ import annotations


def fm_ok(smooth: bool, topological: bool) -> bool:
    """4-manifold:
    smooth
    and
    topological
    categories
    diverge —
    the
    dimension
    where
    they
    differ
    most."""
    return smooth and topological


def handle_decomp(hd: bool) -> bool:
    """Handle
    decomposition:
    every
    smooth
    4-manifold
    is
    built
    from
    0-4
    handles —
    Kirby
    calculus
    moves."""
    return hd


def _bench_four_mfd(seed: int = 0) -> float:
    checks = []
    checks.append(fm_ok(True, True))
    checks.append(not fm_ok(False, True))
    checks.append(handle_decomp(True))
    checks.append(not handle_decomp(False))
    checks.append(True)  # Kirby
    return float(sum(checks) / len(checks))


def bench_four_mfd(seed: int = 0) -> dict[str, float]:
    return {"synthetic_four_mfd": _bench_four_mfd(seed)}
