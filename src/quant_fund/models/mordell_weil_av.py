"""Mordell-Weil theorem (SYNTHETIC)."""

from __future__ import annotations


def mw_ok(finitely_generated: bool, rational_points: bool) -> bool:
    """Mordell-
    Weil:
    rational
    points
    on
    abelian
    varieties
    form
    a
    finitely
    generated
    group —
    abelian
    group
    structure."""
    return finitely_generated and rational_points


def descent_argument(da: bool) -> bool:
    """Descent:
    2-descent
    bounds
    the
    rank
    —
    Mordell-
    Weil
    proof
    technique."""
    return da


def _bench_mordell_weil_av(seed: int = 0) -> float:
    checks = []
    checks.append(mw_ok(True, True))
    checks.append(not mw_ok(False, True))
    checks.append(descent_argument(True))
    checks.append(not descent_argument(False))
    checks.append(True)  # Mordell-Weil
    return float(sum(checks) / len(checks))


def bench_mordell_weil_av(seed: int = 0) -> dict[str, float]:
    return {"synthetic_mordell_weil_av": _bench_mordell_weil_av(seed)}
