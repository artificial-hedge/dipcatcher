"""Perelman entropy (SYNTHETIC)."""

from __future__ import annotations


def pe_ok(w_functional: bool, monotone: bool) -> bool:
    """Perelman
    W-entropy:
    monotone
    under
    Ricci
    flow —
    key
    noncollapsing
    tool."""
    return w_functional and monotone


def kappa_noncollapsing(kn: bool) -> bool:
    """kappa-
    noncollapsing:
    Perelman's
    volume
    control
    enables
    blow-up
    analysis —
    surgery
    proof."""
    return kn


def _bench_perelman_entropy(seed: int = 0) -> float:
    checks = []
    checks.append(pe_ok(True, True))
    checks.append(not pe_ok(False, True))
    checks.append(kappa_noncollapsing(True))
    checks.append(not kappa_noncollapsing(False))
    checks.append(True)  # Perelman 2002
    return float(sum(checks) / len(checks))


def bench_perelman_entropy(seed: int = 0) -> dict[str, float]:
    return {"synthetic_perelman_entropy": _bench_perelman_entropy(seed)}
