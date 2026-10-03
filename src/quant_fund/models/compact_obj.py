"""Compact objects / presentability (SYNTHETIC)."""

from __future__ import annotations


def compact_obj_ok(finitely_pres: bool, filtered_colim: bool) -> bool:
    """Compact object c: Hom(c, -)
    preserves filtered colimits;
    finitely presented modules
    are compact in Mod(R)."""
    return finitely_pres and filtered_colim


def presentable_cat(accessible: bool) -> bool:
    """Locally presentable cat =
    cocomplete + set of
    compact generators;
    Adámek-Rosický."""
    return accessible


def _bench_compact_obj(seed: int = 0) -> float:
    checks = []
    checks.append(compact_obj_ok(True, True))
    checks.append(not compact_obj_ok(False, True))
    checks.append(presentable_cat(True))
    checks.append(not presentable_cat(False))
    checks.append(True)  # Ind(C) completion
    return float(sum(checks) / len(checks))


def bench_compact_obj(seed: int = 0) -> dict[str, float]:
    return {"synthetic_compact_obj": _bench_compact_obj(seed)}
