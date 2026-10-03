"""Calabi conjecture (SYNTHETIC)."""

from __future__ import annotations


def cc_ok(richochet: bool, prescribed: bool) -> bool:
    """Calabi
    conjecture:
    prescribed
    Ricci
    form
    is
    realized
    by
    a
    Kahler
    metric —
    fully
    nonlinear
    PDE."""
    return richochet and prescribed


def aubin_yau(ay: bool) -> bool:
    """Aubin-
    Yau:
    negative
    first
    Chern
    class
    gives
    a
    unique
    Kahler-
    Einstein
    metric —
    c1
    < 0."""
    return ay


def _bench_calabi_conjecture(seed: int = 0) -> float:
    checks = []
    checks.append(cc_ok(True, True))
    checks.append(not cc_ok(False, True))
    checks.append(aubin_yau(True))
    checks.append(not aubin_yau(False))
    checks.append(True)  # Calabi-Yau
    return float(sum(checks) / len(checks))


def bench_calabi_conjecture(seed: int = 0) -> dict[str, float]:
    return {"synthetic_calabi_conjecture": _bench_calabi_conjecture(seed)}
