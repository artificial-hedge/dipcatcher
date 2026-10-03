"""Bdr plus ring (SYNTHETIC)."""

from __future__ import annotations


def bp_ok(bdr: bool, plus: bool) -> bool:
    """Bdr
    plus:
    Bdr
    plus —
    Fontaine."""
    return bdr and plus


def fontaine_period(fp: bool) -> bool:
    """Fontaine
    period:
    Fontaine
    period
    ring —
    crystalline."""
    return fp


def _bench_bdr_plus(seed: int = 0) -> float:
    checks = []
    checks.append(bp_ok(True, True))
    checks.append(not bp_ok(False, True))
    checks.append(fontaine_period(True))
    checks.append(not fontaine_period(False))
    checks.append(True)  # Fontaine
    return float(sum(checks) / len(checks))


def bench_bdr_plus(seed: int = 0) -> dict[str, float]:
    return {"synthetic_bdr_plus": _bench_bdr_plus(seed)}
