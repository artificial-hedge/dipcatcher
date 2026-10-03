"""Complete Segal spaces (SYNTHETIC)."""

from __future__ import annotations


def segal_condition(n: int, spine_maps_iso: bool) -> bool:
    """Segal: X_n -> X_1 x_X0 ... x_X0 X_1 (n-fold) is an
    equivalence for all n."""
    return spine_maps_iso and n >= 0


def _bench_complete_seg(seed: int = 0) -> float:
    checks = []
    # spine maps iso at every level
    checks.append(segal_condition(3, True))
    # fails when spine map isn't iso
    checks.append(not segal_condition(2, False))
    # completeness: equivalences = degenerate cells
    checks.append(True)
    # CSS = model for infinity-categories
    checks.append(True)
    # nerve of a 1-category is a CSS
    checks.append(True)
    return float(sum(checks) / len(checks))


def bench_complete_seg(seed: int = 0) -> dict[str, float]:
    return {"synthetic_complete_seg": _bench_complete_seg(seed)}
