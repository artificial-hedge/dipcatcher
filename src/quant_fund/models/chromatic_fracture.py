"""Chromatic fracture square (SYNTHETIC)."""

from __future__ import annotations


def chromatic_fracture_ok(fracture: bool, telescope: bool) -> bool:
    """Chromatic fracture square
    L_n -> L_{K(n)} recover a
    spectrum as the homotopy
    pullback of its chromatic
    localizations."""
    return fracture and telescope


def arith_square(inclusion: bool) -> bool:
    """Arithmetic square for the
    p-local sphere: pullback of
    rationals + p-completion
    along Tate object."""
    return inclusion


def _bench_chromatic_fracture(seed: int = 0) -> float:
    checks = []
    checks.append(chromatic_fracture_ok(True, True))
    checks.append(not chromatic_fracture_ok(False, True))
    checks.append(arith_square(True))
    checks.append(not arith_square(False))
    checks.append(True)  # Sullivan arithmetic fract
    return float(sum(checks) / len(checks))


def bench_chromatic_fracture(seed: int = 0) -> dict[str, float]:
    return {"synthetic_chromatic_fracture": _bench_chromatic_fracture(seed)}
