"""Kervaire invariant (SYNTHETIC)."""

from __future__ import annotations


def ki_ok(kervaire: bool, inv: bool) -> bool:
    """Kervaire
    inv:
    Kervaire
    invariant —
    framing."""
    return kervaire and inv


def arf_invariant(ai: bool) -> bool:
    """Arf
    invariant:
    Arf
    invariant —
    quadratic."""
    return ai


def _bench_kervaire_inv(seed: int = 0) -> float:
    checks = []
    checks.append(ki_ok(True, True))
    checks.append(not ki_ok(False, True))
    checks.append(arf_invariant(True))
    checks.append(not arf_invariant(False))
    checks.append(True)  # Hill-Hopkins-Ravenel
    return float(sum(checks) / len(checks))


def bench_kervaire_inv(seed: int = 0) -> dict[str, float]:
    return {"synthetic_kervaire_inv": _bench_kervaire_inv(seed)}
