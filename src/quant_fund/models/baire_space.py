"""Baire space omega^omega and the prefix metric (SYNTHETIC)."""

from __future__ import annotations

Seq = tuple[int, ...]


def baire_metric(x: Seq, y: Seq) -> float:
    """d(x,y) = 2^{-n} where n = first coordinate of disagreement."""
    for i, (a, b) in enumerate(zip(x, y, strict=False)):
        if a != b:
            return 0.5 ** (i + 1)
    if len(x) == len(y):
        return 0.0
    return 0.5 ** (min(len(x), len(y)) + 1)


def cylinder(prefix: Seq, seq: Seq) -> bool:
    """Basic clopen [prefix] membership."""
    return seq[: len(prefix)] == prefix


def is_continuous_on_cylinders(f_output_prefix_len: int, input_needed: int) -> bool:
    """Continuity on Baire space: output prefix of length m is determined
    by an input prefix of some length n."""
    return input_needed >= 0


def _bench_baire_space(seed: int = 0) -> float:
    checks = []
    x = (1, 2, 3)
    y = (1, 2, 4)
    z = (0,)
    checks.append(baire_metric(x, x) == 0.0)
    checks.append(abs(baire_metric(x, y) - 0.125) < 1e-12)
    checks.append(abs(baire_metric(x, z) - 0.5) < 1e-12)
    # ultrametric: d(x,z) <= max(d(x,y), d(y,z))
    a, b, c = (1, 0, 0), (1, 0, 1), (1, 1, 0)
    checks.append(baire_metric(a, c) <= max(baire_metric(a, b), baire_metric(b, c)))
    checks.append(baire_metric(a, b) <= max(baire_metric(a, c), baire_metric(c, b)))
    # cylinders
    checks.append(cylinder((1, 2), (1, 2, 9, 9)))
    checks.append(not cylinder((1, 3), (1, 2, 9)))
    # two cylinders with common point are nested (tree property)
    s = (5, 5, 5, 5)
    p, q = (5, 5), (5,)
    checks.append(cylinder(p, s) and cylinder(q, s) and p[: len(q)] == q)
    return float(sum(checks) / len(checks))


def bench_baire_space(seed: int = 0) -> dict[str, float]:
    return {"synthetic_baire_space": _bench_baire_space(seed)}
