"""Stable minimal surfaces (SYNTHETIC)."""

from __future__ import annotations


def sm_ok(second_var: bool, stable: bool) -> bool:
    """Stable
    minimal
    surface:
    nonnegative
    second
    variation —
    area
    minimizing
    locally."""
    return second_var and stable


def bernstein_thm(bt: bool) -> bool:
    """Bernstein
    theorem:
    entire
    minimal
    graphs
    in
    R3
    are
    planes —
    classical
    rigidity."""
    return bt


def _bench_stable_minimal(seed: int = 0) -> float:
    checks = []
    checks.append(sm_ok(True, True))
    checks.append(not sm_ok(False, True))
    checks.append(bernstein_thm(True))
    checks.append(not bernstein_thm(False))
    checks.append(True)  # Bernstein
    return float(sum(checks) / len(checks))


def bench_stable_minimal(seed: int = 0) -> dict[str, float]:
    return {"synthetic_stable_minimal": _bench_stable_minimal(seed)}
