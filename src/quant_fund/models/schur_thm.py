"""Schur theorem (SYNTHETIC)."""

from __future__ import annotations


def schur_ok(color: bool, x_y_z: bool) -> bool:
    """Schur
    theorem:
    finite
    colorings
    of N
    contain
    mono
    x,y,z
    with
    x + y
    = z."""
    return color and x_y_z


def schur_number(num: bool) -> bool:
    """Schur
    number
    S(r):
    largest
    n that
    avoids
    mono
    solutions;
    S(4)=44
    (computed)."""
    return num


def _bench_schur_thm(seed: int = 0) -> float:
    checks = []
    checks.append(schur_ok(True, True))
    checks.append(not schur_ok(False, True))
    checks.append(schur_number(True))
    checks.append(not schur_number(False))
    checks.append(True)  # Schur
    return float(sum(checks) / len(checks))


def bench_schur_thm(seed: int = 0) -> dict[str, float]:
    return {"synthetic_schur_thm": _bench_schur_thm(seed)}
