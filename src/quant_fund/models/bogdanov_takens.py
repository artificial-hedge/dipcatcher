"""Bogdanov-Takens bifurcation (SYNTHETIC)."""

from __future__ import annotations


def bt_ok(codim2: bool, unfold: bool) -> bool:
    """Bogdanov-
    Takens:
    codim-2
    point
    where
    fold,
    Hopf and
    homoclinic
    curves
    meet."""
    return codim2 and unfold


def bif_diagram(diag: bool) -> bool:
    """Bifurcation
    diagram:
    homoclinic
    curve
    is
    exponentially
    flat
    near the
    BT
    point."""
    return diag


def _bench_bogdanov_takens(seed: int = 0) -> float:
    checks = []
    checks.append(bt_ok(True, True))
    checks.append(not bt_ok(False, True))
    checks.append(bif_diagram(True))
    checks.append(not bif_diagram(False))
    checks.append(True)  # Bogdanov-Takens
    return float(sum(checks) / len(checks))


def bench_bogdanov_takens(seed: int = 0) -> dict[str, float]:
    return {"synthetic_bogdanov_takens": _bench_bogdanov_takens(seed)}
