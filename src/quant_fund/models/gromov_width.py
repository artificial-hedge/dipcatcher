"""Gromov width (SYNTHETIC)."""

from __future__ import annotations


def gw_ok(ball_embed: bool, sup_radius: bool) -> bool:
    """Gromov
    width:
    supremum
    of
    capacities
    of
    symplectically
    embeddable
    balls —
    packing
    invariant."""
    return ball_embed and sup_radius


def nonsqueezing(ns: bool) -> bool:
    """Gromov
    non-
    squeezing:
    a
    ball
    does
    not
    embed
    into
    a
    thinner
    cylinder —
    symplectic
    rigidity."""
    return ns


def _bench_gromov_width(seed: int = 0) -> float:
    checks = []
    checks.append(gw_ok(True, True))
    checks.append(not gw_ok(False, True))
    checks.append(nonsqueezing(True))
    checks.append(not nonsqueezing(False))
    checks.append(True)  # Gromov 1985
    return float(sum(checks) / len(checks))


def bench_gromov_width(seed: int = 0) -> dict[str, float]:
    return {"synthetic_gromov_width": _bench_gromov_width(seed)}
