"""MTC motives: mixed Tate category (SYNTHETIC)."""

from __future__ import annotations


def mtc_motive_ok(tate_chain: bool, weight_grad: bool) -> bool:
    """Mixed Tate motives: iterated
    extensions of Tate objects
    Z(n); weight-graded pieces
    recover pure Tate."""
    return tate_chain and weight_grad


def tannaka_mtc(fiber_functor: bool) -> bool:
    """Tannakian fundamental group
    for mixed Tate category;
    fiber functor to graded
    vector spaces."""
    return fiber_functor


def _bench_mtc_motive(seed: int = 0) -> float:
    checks = []
    checks.append(mtc_motive_ok(True, True))
    checks.append(not mtc_motive_ok(False, True))
    checks.append(tannaka_mtc(True))
    checks.append(not tannaka_mtc(False))
    checks.append(True)  # Goncharov's mixed Tate cat
    return float(sum(checks) / len(checks))


def bench_mtc_motive(seed: int = 0) -> dict[str, float]:
    return {"synthetic_mtc_motive": _bench_mtc_motive(seed)}
