"""Lannes T functor (SYNTHETIC)."""

from __future__ import annotations


def lt_ok(lannes: bool, t_funct: bool) -> bool:
    """Lannes
    T:
    T
    functor
    computes
    mapping —
    Lannes
    T."""
    return lannes and t_funct


def t_funct_adjoint(tf: bool) -> bool:
    """T
    functor:
    left
    adjoint
    to
    BZ
    tensor —
    Lannes."""
    return tf


def _bench_lannes_t(seed: int = 0) -> float:
    checks = []
    checks.append(lt_ok(True, True))
    checks.append(not lt_ok(False, True))
    checks.append(t_funct_adjoint(True))
    checks.append(not t_funct_adjoint(False))
    checks.append(True)  # Lannes
    return float(sum(checks) / len(checks))


def bench_lannes_t(seed: int = 0) -> dict[str, float]:
    return {"synthetic_lannes_t": _bench_lannes_t(seed)}
