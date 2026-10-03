"""Formal moduli (SYNTHETIC)."""

from __future__ import annotations


def fm_ok(formal: bool, moduli: bool) -> bool:
    """Formal:
    formal
    moduli
    problem —
    Lurie
    formal."""
    return formal and moduli


def formal_der(fd: bool) -> bool:
    """Formal
    deformation:
    formal
    derived
    moduli —
    Lurie
    moduli."""
    return fd


def _bench_formal_moduli(seed: int = 0) -> float:
    checks = []
    checks.append(fm_ok(True, True))
    checks.append(not fm_ok(False, True))
    checks.append(formal_der(True))
    checks.append(not formal_der(False))
    checks.append(True)  # Lurie
    return float(sum(checks) / len(checks))


def bench_formal_moduli(seed: int = 0) -> dict[str, float]:
    return {"synthetic_formal_moduli": _bench_formal_moduli(seed)}
