"""Beilinson regulators (SYNTHETIC)."""

from __future__ import annotations


def br_ok(beilinson: bool, regulator: bool) -> bool:
    """Beilinson
    regulator:
    Beilinson
    regulator
    map —
    K-theory
    to
    Deligne."""
    return beilinson and regulator


def regulator_map(rm: bool) -> bool:
    """Regulator
    map:
    regulator
    to
    Deligne
    cohomology —
    Borel-type."""
    return rm


def _bench_beilinson_regulator(seed: int = 0) -> float:
    checks = []
    checks.append(br_ok(True, True))
    checks.append(not br_ok(False, True))
    checks.append(regulator_map(True))
    checks.append(not regulator_map(False))
    checks.append(True)  # Beilinson
    return float(sum(checks) / len(checks))


def bench_beilinson_regulator(seed: int = 0) -> dict[str, float]:
    return {"synthetic_beilinson_regulator": _bench_beilinson_regulator(seed)}
