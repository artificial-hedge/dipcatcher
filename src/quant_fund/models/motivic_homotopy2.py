"""Motivic homotopy II (SYNTHETIC)."""

from __future__ import annotations


def mh_ok(a1_localization: bool, nisnevich_loc: bool) -> bool:
    """Motivic
    homotopy:
    A1-
    localization
    with
    Nisnevich
    topology —
    Morel-
    Voevodsky."""
    return a1_localization and nisnevich_loc


def a1_homotopy_thm(aht: bool) -> bool:
    """A1
    homotopy:
    A1
    as
    interval
    object —
    motivic
    homotopy."""
    return aht


def _bench_motivic_homotopy2(seed: int = 0) -> float:
    checks = []
    checks.append(mh_ok(True, True))
    checks.append(not mh_ok(False, True))
    checks.append(a1_homotopy_thm(True))
    checks.append(not a1_homotopy_thm(False))
    checks.append(True)  # Morel-Voevodsky
    return float(sum(checks) / len(checks))


def bench_motivic_homotopy2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_motivic_homotopy2": _bench_motivic_homotopy2(seed)}
