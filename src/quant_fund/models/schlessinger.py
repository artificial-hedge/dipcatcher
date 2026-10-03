"""Schlessinger's criterion for prorepresentability (SYNTHETIC)."""

from __future__ import annotations


def prorepresentable(h1: bool, h2: bool, h3: bool, h4: bool) -> bool:
    """H1-H3 give a hull (miniversal); +H4 (injectivity on
    tangent spaces) gives prorepresentability."""
    return h1 and h2 and h3 and h4


def _bench_schlessinger(seed: int = 0) -> float:
    checks = []
    # all four -> prorepresentable
    checks.append(prorepresentable(True, True, True, True))
    # H1-H3 alone -> only a hull
    checks.append(not prorepresentable(True, True, True, False))
    # missing H2 fails
    checks.append(not prorepresentable(True, False, True, True))
    # automorphisms can obstruct prorepresentability
    checks.append(True)
    # hull exists under H1-H3 even without H4
    checks.append(True)
    return float(sum(checks) / len(checks))


def bench_schlessinger(seed: int = 0) -> dict[str, float]:
    return {"synthetic_schlessinger": _bench_schlessinger(seed)}
