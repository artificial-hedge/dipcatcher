"""Arithmetic D-modules (SYNTHETIC)."""

from __future__ import annotations


def arithmetic_level(m_level: int, formal_sch: bool) -> bool:
    """D^(m) on formal scheme has level m
    differential operators; m -> infty gives
    the full D-module."""
    return m_level >= 0 and formal_sch


def char_p_dm(weil_alg: bool) -> bool:
    """In characteristic p, D_X is the Weyl
    algebra with divided-power structure."""
    return weil_alg


def _bench_arithmetic_dm(seed: int = 0) -> float:
    checks = []
    checks.append(arithmetic_level(0, True))
    checks.append(arithmetic_level(2, True))
    checks.append(not arithmetic_level(-1, True))
    checks.append(char_p_dm(True))
    checks.append(not char_p_dm(False))
    return float(sum(checks) / len(checks))


def bench_arithmetic_dm(seed: int = 0) -> dict[str, float]:
    return {"synthetic_arithmetic_dm": _bench_arithmetic_dm(seed)}
