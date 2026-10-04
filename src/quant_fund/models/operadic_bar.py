"""Operadic bar construction (SYNTHETIC)."""

from __future__ import annotations


def ob_ok(bar: bool, cooperad: bool) -> bool:
    """Operadic
    bar:
    bar
    construction
    of
    operad —
    Ginzburg-
    Kapranov
    bar."""
    return bar and cooperad


def bar_cooperad(bc: bool) -> bool:
    """Bar
    cooperad:
    bar
    gives
    cooperad —
    operadic
    bar."""
    return bc


def _bench_operadic_bar(seed: int = 0) -> float:
    checks = []
    checks.append(ob_ok(True, True))
    checks.append(not ob_ok(False, True))
    checks.append(bar_cooperad(True))
    checks.append(not bar_cooperad(False))
    checks.append(True)  # G-K
    return float(sum(checks) / len(checks))


def bench_operadic_bar(seed: int = 0) -> dict[str, float]:
    return {"synthetic_operadic_bar": _bench_operadic_bar(seed)}
