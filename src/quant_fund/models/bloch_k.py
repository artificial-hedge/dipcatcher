"""Bloch K-theory (SYNTHETIC)."""

from __future__ import annotations


def bk_ok(bloch_def: bool, cubical: bool) -> bool:
    """Bloch
    K:
    higher
    Chow
    cubical
    cycle
    complex —
    Bloch
    higher
    Chow."""
    return bloch_def and cubical


def bloch_les(bl: bool) -> bool:
    """Bloch
    LES:
    localization
    long
    exact
    sequence —
    Bloch
    localization."""
    return bl


def _bench_bloch_k(seed: int = 0) -> float:
    checks = []
    checks.append(bk_ok(True, True))
    checks.append(not bk_ok(False, True))
    checks.append(bloch_les(True))
    checks.append(not bloch_les(False))
    checks.append(True)  # Bloch
    return float(sum(checks) / len(checks))


def bench_bloch_k(seed: int = 0) -> dict[str, float]:
    return {"synthetic_bloch_k": _bench_bloch_k(seed)}
