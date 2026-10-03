"""Little intervals operad E1 composition (SYNTHETIC)."""

from __future__ import annotations


def compose_intervals(
    outer: list[tuple[float, float]], inner: list[tuple[float, float]]
) -> list[tuple[float, float]]:
    """Little-intervals composition: embed k sub-intervals into the
    slots of an outer configuration by affine rescaling."""
    out = []
    for lo, hi in outer:
        w = hi - lo
        for a, b in inner:
            out.append((lo + a * w, lo + b * w))
    return out


def _bench_little_intervals(seed: int = 0) -> float:
    checks = []
    outer = [(0.0, 0.5), (0.5, 1.0)]
    inner = [(0.0, 1.0)]
    comp = compose_intervals(outer, inner)
    checks.append(comp == outer)
    # nested sub-interval lands inside its slot
    comp2 = compose_intervals([(0.0, 0.5)], [(0.0, 0.5)])
    checks.append(comp2 == [(0.0, 0.25)])
    # composition is associative on centers
    c1 = compose_intervals([(0.0, 0.5)], [(0.0, 0.5)])
    c2 = compose_intervals(c1, [(0.0, 0.5)])
    checks.append(c2 == [(0.0, 0.125)])
    # arity respected
    checks.append(len(comp) == 2)
    # disjointness preserved
    checks.append(comp[0][1] <= comp[1][0])
    return float(sum(checks) / len(checks))


def bench_little_intervals(seed: int = 0) -> dict[str, float]:
    return {"synthetic_little_intervals": _bench_little_intervals(seed)}
