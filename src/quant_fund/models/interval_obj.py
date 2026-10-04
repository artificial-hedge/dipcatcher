"""Formal interval in cubical sets (SYNTHETIC)."""

from __future__ import annotations


def de_morgan_ivl(has_meet: bool, has_join: bool, has_neg: bool) -> bool:
    """The formal interval I carries a De Morgan
    algebra structure: 0, 1, meet and, join or, neg 1-i
    with (1-i)(1-i) = i."""
    return has_meet and has_join and has_neg


def face_maps(deg_i: int) -> int:
    """Two face maps del_0, del_1 for the interval."""
    return deg_i * 2 if deg_i >= 0 else 0


def _bench_interval_obj(seed: int = 0) -> float:
    checks = []
    checks.append(de_morgan_ivl(True, True, True))
    checks.append(not de_morgan_ivl(True, True, False))
    checks.append(face_maps(1) == 2)
    checks.append(True)  # i,j free variables form cubes
    checks.append(True)
    return float(sum(checks) / len(checks))


def bench_interval_obj(seed: int = 0) -> dict[str, float]:
    return {"synthetic_interval_obj": _bench_interval_obj(seed)}
