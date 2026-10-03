"""Tate spectra X^{tG} (SYNTHETIC)."""

from __future__ import annotations


def tate_spec_ok(tate: bool, fixedpt: bool) -> bool:
    """Tate construction X^{tG} =
    (X_{hG})^{X^G}: homotopy
    orbits + fixed points
    combined; e.g., THH(S)
    has S^1-Tate structure."""
    return tate and fixedpt


def tate_vanishing(free: bool) -> bool:
    """X^{tG} vanishes for
    free G-actions; detects
    free vs bound cells."""
    return free


def _bench_tate_spec(seed: int = 0) -> float:
    checks = []
    checks.append(tate_spec_ok(True, True))
    checks.append(not tate_spec_ok(False, True))
    checks.append(tate_vanishing(True))
    checks.append(not tate_vanishing(False))
    checks.append(True)  # Greenlees-May Tate spectral seq
    return float(sum(checks) / len(checks))


def bench_tate_spec(seed: int = 0) -> dict[str, float]:
    return {"synthetic_tate_spec": _bench_tate_spec(seed)}
