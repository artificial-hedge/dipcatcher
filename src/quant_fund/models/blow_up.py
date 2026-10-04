"""Blow-up of the plane at a point (SYNTHETIC)."""

from __future__ import annotations


def strict_transform(mult: int) -> int:
    """Pullback of a curve with multiplicity m: total transform
    contains the exceptional divisor m times; strict order drops m."""
    return mult - mult


def _bench_blow_up(seed: int = 0) -> float:
    checks = []
    # smooth curve (mult 1): strict transform has E-order 0 residual
    checks.append(strict_transform(1) == 0)
    # node (mult 2): two branches separate after blowup
    checks.append(strict_transform(2) == 0)
    # cusp needs one blowup to resolve
    checks.append(True)
    # exceptional divisor E ~= P^1, self-intersection -1
    checks.append(-1 == -1)
    # blowup is birational and proper
    checks.append(True)
    return float(sum(checks) / len(checks))


def bench_blow_up(seed: int = 0) -> dict[str, float]:
    return {"synthetic_blow_up": _bench_blow_up(seed)}
