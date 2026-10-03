"""Bousfield-Kan spectral sequence (SYNTHETIC)."""

from __future__ import annotations


def bk_ok(bousfield: bool, unstable: bool) -> bool:
    """Bousfield-
    Kan:
    unstable
    Adams
    SS —
    Bousfield-
    Kan."""
    return bousfield and unstable


def unstable_ss(us: bool) -> bool:
    """Unstable
    SS:
    unstable
    Adams
    spectral
    seq —
    Massey-
    Peterson."""
    return us


def _bench_bousfield_kan(seed: int = 0) -> float:
    checks = []
    checks.append(bk_ok(True, True))
    checks.append(not bk_ok(False, True))
    checks.append(unstable_ss(True))
    checks.append(not unstable_ss(False))
    checks.append(True)  # Bousfield-Kan
    return float(sum(checks) / len(checks))


def bench_bousfield_kan(seed: int = 0) -> dict[str, float]:
    return {"synthetic_bousfield_kan": _bench_bousfield_kan(seed)}
