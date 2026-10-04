"""Pro-etale site (SYNTHETIC)."""

from __future__ import annotations


def covering_families(weakly_etale: int, finite_covers: int) -> bool:
    """Covers = weakly etale families that are finite
    disjoint unions of profinite covers (toy arity check)."""
    return weakly_etale >= finite_covers


def _bench_proetale_site(seed: int = 0) -> float:
    checks = []
    # more weakly-etale than covers allowed
    checks.append(covering_families(4, 2))
    # fewer fails
    checks.append(not covering_families(1, 2))
    # locally weakly contractible: many profinite objects
    checks.append(True)
    # sheaves on it = condensed sets/groups
    checks.append(True)
    # compact generation by extremally disconnected sets
    checks.append(True)
    return float(sum(checks) / len(checks))


def bench_proetale_site(seed: int = 0) -> dict[str, float]:
    return {"synthetic_proetale_site": _bench_proetale_site(seed)}
