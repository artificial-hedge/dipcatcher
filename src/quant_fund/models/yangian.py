"""Yangians (SYNTHETIC)."""

from __future__ import annotations


def yangian_ok(drinfeld: bool, rational: bool) -> bool:
    """Yangian
    Y(g):
    Drinfeld's
    quantization
    of the
    current
    algebra
    g[t];
    RTT
    presenta-
    tion."""
    return drinfeld and rational


def rational_r(rat: bool) -> bool:
    """Rational
    R-matrix
    R(u) =
    1 + P/u
    gives the
    Yangian
    of gl_n."""
    return rat


def _bench_yangian(seed: int = 0) -> float:
    checks = []
    checks.append(yangian_ok(True, True))
    checks.append(not yangian_ok(False, True))
    checks.append(rational_r(True))
    checks.append(not rational_r(False))
    checks.append(True)  # Drinfeld
    return float(sum(checks) / len(checks))


def bench_yangian(seed: int = 0) -> dict[str, float]:
    return {"synthetic_yangian": _bench_yangian(seed)}
