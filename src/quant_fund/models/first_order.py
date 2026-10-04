"""First-order deformation (SYNTHETIC)."""

from __future__ import annotations


def fo_ok(first_order: bool, square: bool) -> bool:
    """First
    order:
    first-
    order
    deformation —
    dual
    numbers."""
    return first_order and square


def dual_numbers(dn: bool) -> bool:
    """Dual
    numbers:
    first-
    order
    over
    dual
    numbers —
    dual
    numbers."""
    return dn


def _bench_first_order(seed: int = 0) -> float:
    checks = []
    checks.append(fo_ok(True, True))
    checks.append(not fo_ok(False, True))
    checks.append(dual_numbers(True))
    checks.append(not dual_numbers(False))
    checks.append(True)  # dual numbers
    return float(sum(checks) / len(checks))


def bench_first_order(seed: int = 0) -> dict[str, float]:
    return {"synthetic_first_order": _bench_first_order(seed)}
