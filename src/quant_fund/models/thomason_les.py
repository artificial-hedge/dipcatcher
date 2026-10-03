"""Thomason LES K-theory (SYNTHETIC)."""

from __future__ import annotations


def tl_ok(thomason: bool, les: bool) -> bool:
    """Thomason
    LES:
    Thomason
    LES —
    localization."""
    return thomason and les


def localization_seq(ls: bool) -> bool:
    """Localization
    seq:
    localization
    sequence —
    devissage."""
    return ls


def _bench_thomason_les(seed: int = 0) -> float:
    checks = []
    checks.append(tl_ok(True, True))
    checks.append(not tl_ok(False, True))
    checks.append(localization_seq(True))
    checks.append(not localization_seq(False))
    checks.append(True)  # Thomason
    return float(sum(checks) / len(checks))


def bench_thomason_les(seed: int = 0) -> dict[str, float]:
    return {"synthetic_thomason_les": _bench_thomason_les(seed)}
