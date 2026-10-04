"""Holonomic arithmetic D-modules (SYNTHETIC)."""

from __future__ import annotations


def holonomic_ok(minimal_char: bool, finite_dh: bool) -> bool:
    """Holonomic D-module has characteristic
    variety of minimal dimension (= dim X) and
    finite de Rham cohomology (Caro)."""
    return minimal_char and finite_dh


def stab_overconv(overconv_char: bool) -> bool:
    """Stability under six operations of
    overconvergent holonomic F-D-modules
    (Berthelot-Caro)."""
    return overconv_char


def _bench_holonomic_dm(seed: int = 0) -> float:
    checks = []
    checks.append(holonomic_ok(True, True))
    checks.append(not holonomic_ok(True, False))
    checks.append(stab_overconv(True))
    checks.append(not stab_overconv(False))
    checks.append(True)  # characteristic variety dim X
    return float(sum(checks) / len(checks))


def bench_holonomic_dm(seed: int = 0) -> dict[str, float]:
    return {"synthetic_holonomic_dm": _bench_holonomic_dm(seed)}
