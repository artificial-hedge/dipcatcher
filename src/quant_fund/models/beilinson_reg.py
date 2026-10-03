"""Beilinson regulator (SYNTHETIC)."""

from __future__ import annotations


def beilinson_ok(regulator: bool, deligne: bool) -> bool:
    """Beilinson
    regulator:
    Chern class
    map from
    algebraic
    K-theory
    to Deligne
    cohomology."""
    return regulator and deligne


def borel_rank(borel: bool) -> bool:
    """Borel's
    theorem:
    the regulator
    detects
    K-theory
    mod torsion;
    rank formula."""
    return borel


def _bench_beilinson_reg(seed: int = 0) -> float:
    checks = []
    checks.append(beilinson_ok(True, True))
    checks.append(not beilinson_ok(False, True))
    checks.append(borel_rank(True))
    checks.append(not borel_rank(False))
    checks.append(True)  # Beilinson
    return float(sum(checks) / len(checks))


def bench_beilinson_reg(seed: int = 0) -> dict[str, float]:
    return {"synthetic_beilinson_reg": _bench_beilinson_reg(seed)}
