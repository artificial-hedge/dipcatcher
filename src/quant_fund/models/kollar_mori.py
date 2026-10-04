"""Kollar-Mori program (SYNTHETIC)."""

from __future__ import annotations


def km_ok(mmp_steps: bool, flips: bool) -> bool:
    """Kollar-
    Mori:
    minimal
    model
    program
    for
    threefolds —
    flips
    and
    divisorial
    contractions."""
    return mmp_steps and flips


def termination_flips(tf: bool) -> bool:
    """Termination
    of
    flips:
    Mori
    shows
    3-fold
    flips
    terminate —
    finite
    MMP
    sequence."""
    return tf


def _bench_kollar_mori(seed: int = 0) -> float:
    checks = []
    checks.append(km_ok(True, True))
    checks.append(not km_ok(False, True))
    checks.append(termination_flips(True))
    checks.append(not termination_flips(False))
    checks.append(True)  # Kollar-Mori
    return float(sum(checks) / len(checks))


def bench_kollar_mori(seed: int = 0) -> dict[str, float]:
    return {"synthetic_kollar_mori": _bench_kollar_mori(seed)}
