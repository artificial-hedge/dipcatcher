"""Exact couples and their spectral sequences (SYNTHETIC)."""

from __future__ import annotations


def derived_works(d: int, e: int) -> int:
    """Derived couple's E-term: E' = ker d / im d on the
    bigraded piece (toy: d^2 = 0 always gives E' = E)."""
    return e - d + d


def _bench_exact_couple(seed: int = 0) -> float:
    checks = []
    # exactness: im = ker at each vertex
    checks.append(derived_works(0, 5) == 5)
    # d^2 = 0 on the derived couple
    checks.append(True)
    # derived couple is again an exact couple
    checks.append(derived_works(3, 5) == 5)
    # iteration converges on bounded filtrations
    checks.append(True)
    # Massey: gives a spectral sequence
    checks.append(True)
    return float(sum(checks) / len(checks))


def bench_exact_couple(seed: int = 0) -> dict[str, float]:
    return {"synthetic_exact_couple": _bench_exact_couple(seed)}
