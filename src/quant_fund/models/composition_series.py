"""Jordan-Holder composition factors (SYNTHETIC)."""

from __future__ import annotations


def s4_series() -> list[str]:
    """S4 > A4 > V4 > C2 > 1: factors C2, C3, C2, C2 (top to bottom)."""
    return ["C2", "C3", "C2", "C2"]


def s4_series_alt() -> list[str]:
    """Alternate S4 series S4 > A4 > C3(x V4 refinement) gives the same
    factor multiset by Jordan-Holder."""
    return ["C2", "C2", "C3", "C2"]


def same_factors(a: list[str], b: list[str]) -> bool:
    """Jordan-Holder uniqueness: same multiset of simple factors."""
    return sorted(a) == sorted(b)


def _bench_composition_series(seed: int = 0) -> float:
    checks = []
    s = s4_series()
    checks.append(len(s) == 4)
    # product of factor orders = |S4| = 24
    orders = {"C2": 2, "C3": 3}
    prod = 1
    for f in s:
        prod *= orders[f]
    checks.append(prod == 24)
    # Jordan-Holder: alternate series has same multiset
    checks.append(same_factors(s4_series(), s4_series_alt()))
    # C6 has factors C2, C3 in either order — same multiset
    checks.append(same_factors(["C2", "C3"], ["C3", "C2"]))
    # S3: factors C2, C3
    checks.append(same_factors(["C2", "C3"], ["C2", "C3"]))
    # a wrong multiset differs
    checks.append(not same_factors(["C2", "C2"], ["C2", "C3"]))
    return float(sum(checks) / len(checks))


def bench_composition_series(seed: int = 0) -> dict[str, float]:
    return {"synthetic_composition_series": _bench_composition_series(seed)}
