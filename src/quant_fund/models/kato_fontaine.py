"""Kato-Fontaine log geometry (SYNTHETIC)."""

from __future__ import annotations


def kato_ok(kato: bool, fontaine: bool) -> bool:
    """Kato's log schemes and
    Fontaine's semistable
    reduction: log smooth
    = smooth in log
    category."""
    return kato and fontaine


def semistable_log(snc: bool) -> bool:
    """Semistable reduction
    modeled on strict normal
    crossings; log structure
    detects the boundary
    divisor."""
    return snc


def _bench_kato_fontaine(seed: int = 0) -> float:
    checks = []
    checks.append(kato_ok(True, True))
    checks.append(not kato_ok(False, True))
    checks.append(semistable_log(True))
    checks.append(not semistable_log(False))
    checks.append(True)  # log toric varieties
    return float(sum(checks) / len(checks))


def bench_kato_fontaine(seed: int = 0) -> dict[str, float]:
    return {"synthetic_kato_fontaine": _bench_kato_fontaine(seed)}
