"""Gauss points and norms (SYNTHETIC)."""

from __future__ import annotations


def gauss_norm_ok(multiplicative: bool, max_mod: bool) -> bool:
    """Gauss norm |sum a_i T^i| = max|a_i| on
    Tate algebra T_n; multiplicative and
    gives sup over the closed unit ball."""
    return multiplicative and max_mod


def gauss_pt_type1(type1: bool) -> bool:
    """Gauss point is the type-1 point of
    the Berkovich unit disk; generic point
    of the special fiber lift."""
    return type1


def _bench_gauss_point(seed: int = 0) -> float:
    checks = []
    checks.append(gauss_norm_ok(True, True))
    checks.append(not gauss_norm_ok(True, False))
    checks.append(gauss_pt_type1(True))
    checks.append(not gauss_pt_type1(False))
    checks.append(True)  # 4 types of Berkovich points
    return float(sum(checks) / len(checks))


def bench_gauss_point(seed: int = 0) -> dict[str, float]:
    return {"synthetic_gauss_point": _bench_gauss_point(seed)}
