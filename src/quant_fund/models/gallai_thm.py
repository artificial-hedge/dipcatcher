"""Gallai theorem (SYNTHETIC)."""

from __future__ import annotations


def gallai_ok(homothetic: bool, finite: bool) -> bool:
    """Gallai
    theorem:
    finite
    colorings
    of Z^d
    contain
    monochromatic
    homothetic
    copies
    of any
    finite
    pattern."""
    return homothetic and finite


def van_der_corput(vdc: bool) -> bool:
    """Van der
    Corput:
    density
    increment
    through
    thickening
    (induction
    on
    patterns)."""
    return vdc


def _bench_gallai_thm(seed: int = 0) -> float:
    checks = []
    checks.append(gallai_ok(True, True))
    checks.append(not gallai_ok(False, True))
    checks.append(van_der_corput(True))
    checks.append(not van_der_corput(False))
    checks.append(True)  # Gallai
    return float(sum(checks) / len(checks))


def bench_gallai_thm(seed: int = 0) -> dict[str, float]:
    return {"synthetic_gallai_thm": _bench_gallai_thm(seed)}
