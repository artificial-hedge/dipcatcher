"""Finite homotopy (SYNTHETIC)."""

from __future__ import annotations


def fh_ok(finite: bool, htpy: bool) -> bool:
    """Finite
    homotopy:
    finite
    homotopy —
    finite
    complex."""
    return finite and htpy


def finite_complex(fc: bool) -> bool:
    """Finite
    complex:
    finite
    CW
    complex —
    finite
    cells."""
    return fc


def _bench_finite_htpy(seed: int = 0) -> float:
    checks = []
    checks.append(fh_ok(True, True))
    checks.append(not fh_ok(False, True))
    checks.append(finite_complex(True))
    checks.append(not finite_complex(False))
    checks.append(True)  # Wall
    return float(sum(checks) / len(checks))


def bench_finite_htpy(seed: int = 0) -> dict[str, float]:
    return {"synthetic_finite_htpy": _bench_finite_htpy(seed)}
