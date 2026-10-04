"""Teichmuller lift (SYNTHETIC)."""

from __future__ import annotations


def wt_ok(teichmuller: bool, lift: bool) -> bool:
    """Teichmuller:
    Teichmuller
    lift
    of
    coordinates —
    Teichmuller
    representative."""
    return teichmuller and lift


def teich_multiplicative(tm: bool) -> bool:
    """Teichmuller
    multiplicative:
    the
    Teichmuller
    lift
    is
    multiplicative —
    representative."""
    return tm


def _bench_witt_teich(seed: int = 0) -> float:
    checks = []
    checks.append(wt_ok(True, True))
    checks.append(not wt_ok(False, True))
    checks.append(teich_multiplicative(True))
    checks.append(not teich_multiplicative(False))
    checks.append(True)  # Teichmuller
    return float(sum(checks) / len(checks))


def bench_witt_teich(seed: int = 0) -> dict[str, float]:
    return {"synthetic_witt_teich": _bench_witt_teich(seed)}
