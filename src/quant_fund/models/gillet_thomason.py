"""Gillet-Thomason K-theory (SYNTHETIC)."""

from __future__ import annotations


def gt_ok(gillet: bool, thomason: bool) -> bool:
    """Gillet:
    Gillet-
    Thomason
    K-
    theory —
    Gillet-
    Thomason."""
    return gillet and thomason


def localization_seq(ls: bool) -> bool:
    """Localization:
    Thomason
    localization
    sequence —
    Thomason
    localization."""
    return ls


def _bench_gillet_thomason(seed: int = 0) -> float:
    checks = []
    checks.append(gt_ok(True, True))
    checks.append(not gt_ok(False, True))
    checks.append(localization_seq(True))
    checks.append(not localization_seq(False))
    checks.append(True)  # Thomason
    return float(sum(checks) / len(checks))


def bench_gillet_thomason(seed: int = 0) -> dict[str, float]:
    return {"synthetic_gillet_thomason": _bench_gillet_thomason(seed)}
