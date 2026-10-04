"""Mori birational geometry (SYNTHETIC)."""

from __future__ import annotations


def mb_ok(mori: bool, birational: bool) -> bool:
    """Mori:
    Mori
    birational
    geometry —
    Mori
    program."""
    return mori and birational


def mori_theory(mt: bool) -> bool:
    """Mori
    theory:
    minimal
    model
    program
    theory —
    Mori."""
    return mt


def _bench_mori_bir(seed: int = 0) -> float:
    checks = []
    checks.append(mb_ok(True, True))
    checks.append(not mb_ok(False, True))
    checks.append(mori_theory(True))
    checks.append(not mori_theory(False))
    checks.append(True)  # Mori
    return float(sum(checks) / len(checks))


def bench_mori_bir(seed: int = 0) -> dict[str, float]:
    return {"synthetic_mori_bir": _bench_mori_bir(seed)}
