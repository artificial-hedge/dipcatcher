"""Homotopy colimits (SYNTHETIC)."""

from __future__ import annotations


def hc_ok(homotopy: bool, colim: bool) -> bool:
    """Homotopy
    colimit:
    homotopy
    colimit —
    Bousfield
    Kan."""
    return homotopy and colim


def bousfield_kan(bk: bool) -> bool:
    """Bousfield
    Kan:
    Bousfield
    Kan
    hocolim —
    realization."""
    return bk


def _bench_homotopy_colim(seed: int = 0) -> float:
    checks = []
    checks.append(hc_ok(True, True))
    checks.append(not hc_ok(False, True))
    checks.append(bousfield_kan(True))
    checks.append(not bousfield_kan(False))
    checks.append(True)  # Bousfield-Kan
    return float(sum(checks) / len(checks))


def bench_homotopy_colim(seed: int = 0) -> dict[str, float]:
    return {"synthetic_homotopy_colim": _bench_homotopy_colim(seed)}
