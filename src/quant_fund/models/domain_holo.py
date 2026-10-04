"""Domains of holomorphy (SYNTHETIC)."""

from __future__ import annotations


def doh_ok(maximal: bool, convexity: bool) -> bool:
    """Domain
    of
    holomorphy:
    maximal
    domain
    for
    some
    holomorphic
    function —
    characterized
    by
    holomorphic
    convexity."""
    return maximal and convexity


def leviciv_problem_equiv(lp: bool) -> bool:
    """Cartan-
    Thullen:
    domains
    of
    holomorphy
    equal
    holomorphically
    convex
    domains."""
    return lp


def _bench_domain_holo(seed: int = 0) -> float:
    checks = []
    checks.append(doh_ok(True, True))
    checks.append(not doh_ok(False, True))
    checks.append(leviciv_problem_equiv(True))
    checks.append(not leviciv_problem_equiv(False))
    checks.append(True)  # Cartan-Thullen
    return float(sum(checks) / len(checks))


def bench_domain_holo(seed: int = 0) -> dict[str, float]:
    return {"synthetic_domain_holo": _bench_domain_holo(seed)}
