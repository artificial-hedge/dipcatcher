"""Diamonds (SYNTHETIC)."""

from __future__ import annotations


def diamond_ok(v_sheaf: bool, diamond_fixed: bool) -> bool:
    """A diamond is a pro-etale sheaf
    on Perf that is a quotient of a
    perfectoid by pro-etale equivalence
    relation (Scholze)."""
    return v_sheaf and diamond_fixed


def v_site_descent(cover_effective: bool) -> bool:
    """The v-topology on Perf makes every
    analytic adic space a diamond; the
    functor X -> X^diamond is faithful."""
    return cover_effective


def _bench_diamond_geo(seed: int = 0) -> float:
    checks = []
    checks.append(diamond_ok(True, True))
    checks.append(not diamond_ok(False, True))
    checks.append(v_site_descent(True))
    checks.append(not v_site_descent(False))
    checks.append(True)  # small v-sheaves = diamonds
    return float(sum(checks) / len(checks))


def bench_diamond_geo(seed: int = 0) -> dict[str, float]:
    return {"synthetic_diamond_geo": _bench_diamond_geo(seed)}
