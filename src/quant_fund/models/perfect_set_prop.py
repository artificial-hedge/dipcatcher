"""Perfect set property and Cantor-Bendixson derivative (SYNTHETIC)."""

from __future__ import annotations

Seq = tuple[int, ...]


def is_isolated(x: Seq, subset: list[Seq] | tuple[Seq, ...], radius: float) -> bool:
    """x isolated in subset iff no other member shares a long prefix."""
    from quant_fund.models.baire_space import baire_metric

    return all(x == y or baire_metric(x, y) > radius for y in subset)


def cantor_bendixson_deriv(subset: list[Seq] | tuple[Seq, ...], radius: float) -> list[Seq]:
    """Remove isolated points."""
    return [x for x in subset if not is_isolated(x, subset, radius)]


def _bench_perfect_set_prop(seed: int = 0) -> float:
    checks = []
    # full finite "Cantor-like" set: all 2^4 points, no point isolated
    # at radius smaller than closest pair distance 1/16
    from itertools import product

    full = list(product((0, 1), repeat=4))
    # points differing only in last coord are at distance 2^-4 = 0.0625;
    # radius 0.07 sees them as neighbors -> nothing isolated
    checks.append(cantor_bendixson_deriv(full, 0.07) == full)
    # a singleton is isolated -> derivative empties it
    checks.append(cantor_bendixson_deriv([(0, 0, 0, 0)], 0.07) == [])
    # two points far apart (first coordinate differs, d=0.5 > 0.07) are
    # mutually isolated -> both removed
    checks.append(cantor_bendixson_deriv([(0, 0, 0, 0), (1, 1, 1, 1)], 0.07) == [])
    # pair at distance 2^-6 ~ 0.0156 (sharing 5-prefix on length-6) is
    # inside radius 0.07 -> survives as a cluster
    pts: list[tuple[int, ...]] = [(0, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 1)]
    checks.append(cantor_bendixson_deriv(pts, 0.07) == pts)
    # perfect set property dichotomy holds on these toys: empty or "fat"
    checks.append(len(cantor_bendixson_deriv(full, 0.07)) == 16)
    return float(sum(checks) / len(checks))


def bench_perfect_set_prop(seed: int = 0) -> dict[str, float]:
    return {"synthetic_perfect_set_prop": _bench_perfect_set_prop(seed)}
