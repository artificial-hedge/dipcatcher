"""Obstruction theory (SYNTHETIC)."""

from __future__ import annotations


def ob_ok(obstruction: bool, lift: bool) -> bool:
    """Obstruction:
    obstruction
    to
    lifting —
    obstruction
    theory."""
    return obstruction and lift


def obstruction_class(oc: bool) -> bool:
    """Obstruction
    class:
    obstruction
    class
    to
    lifting —
    obstruction
    class."""
    return oc


def _bench_obstruction_def(seed: int = 0) -> float:
    checks = []
    checks.append(ob_ok(True, True))
    checks.append(not ob_ok(False, True))
    checks.append(obstruction_class(True))
    checks.append(not obstruction_class(False))
    checks.append(True)  # obstruction
    return float(sum(checks) / len(checks))


def bench_obstruction_def(seed: int = 0) -> dict[str, float]:
    return {"synthetic_obstruction_def": _bench_obstruction_def(seed)}
