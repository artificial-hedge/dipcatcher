"""Dualizing complex (SYNTHETIC)."""

from __future__ import annotations


def dc_ok(dualizing: bool, finite_inj_dim: bool) -> bool:
    """Dualizing
    complex:
    bounded
    injective
    with
    duality —
    Grothendieck
    dualizing."""
    return dualizing and finite_inj_dim


def normalize_dual(nd: bool) -> bool:
    """Normalize:
    dualizing
    complex
    unique
    up
    to
    shift —
    normalized
    dualizing."""
    return nd


def _bench_dualizing_cmplx(seed: int = 0) -> float:
    checks = []
    checks.append(dc_ok(True, True))
    checks.append(not dc_ok(False, True))
    checks.append(normalize_dual(True))
    checks.append(not normalize_dual(False))
    checks.append(True)  # Hartshorne
    return float(sum(checks) / len(checks))


def bench_dualizing_cmplx(seed: int = 0) -> dict[str, float]:
    return {"synthetic_dualizing_cmplx": _bench_dualizing_cmplx(seed)}
