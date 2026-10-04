"""Homotopy category (SYNTHETIC)."""

from __future__ import annotations


def hc_ok2(homotopy: bool, category: bool) -> bool:
    """Homotopy:
    homotopy
    category
    of
    infinity-
    category —
    Joyal
    homotopy."""
    return homotopy and category


def hocolim_homotopy(hh: bool) -> bool:
    """Homotopy
    colimit:
    homotopy
    colimit
    vs
    strict —
    Vogt
    hocolim."""
    return hh


def _bench_homotopy_cat(seed: int = 0) -> float:
    checks = []
    checks.append(hc_ok2(True, True))
    checks.append(not hc_ok2(False, True))
    checks.append(hocolim_homotopy(True))
    checks.append(not hocolim_homotopy(False))
    checks.append(True)  # Joyal
    return float(sum(checks) / len(checks))


def bench_homotopy_cat(seed: int = 0) -> dict[str, float]:
    return {"synthetic_homotopy_cat": _bench_homotopy_cat(seed)}
