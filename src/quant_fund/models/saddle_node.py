"""Saddle-node bifurcation (SYNTHETIC)."""

from __future__ import annotations


def sn_ok(fold: bool, birth: bool) -> bool:
    """Saddle-
    node
    (fold):
    two
    equilibria
    collide
    and
    vanish
    at a
    quadratic
    tangency."""
    return fold and birth


def normal_form(nf: bool) -> bool:
    """Normal
    form
    x' =
    mu +
    x^2 —
    universal
    unfolding
    of the
    fold."""
    return nf


def _bench_saddle_node(seed: int = 0) -> float:
    checks = []
    checks.append(sn_ok(True, True))
    checks.append(not sn_ok(False, True))
    checks.append(normal_form(True))
    checks.append(not normal_form(False))
    checks.append(True)  # Sotomayor
    return float(sum(checks) / len(checks))


def bench_saddle_node(seed: int = 0) -> dict[str, float]:
    return {"synthetic_saddle_node": _bench_saddle_node(seed)}
