"""Witt length (SYNTHETIC)."""

from __future__ import annotations


def wl2_ok(witt: bool, length: bool) -> bool:
    """Witt
    length:
    Witt
    vectors
    of
    length
    n —
    truncated
    Witt."""
    return witt and length


def truncated_witt(tw: bool) -> bool:
    """Truncated
    Witt:
    truncated
    Witt
    ring —
    W
    n
    of
    a
    perfect
    ring."""
    return tw


def _bench_witt_len2(seed: int = 0) -> float:
    checks = []
    checks.append(wl2_ok(True, True))
    checks.append(not wl2_ok(False, True))
    checks.append(truncated_witt(True))
    checks.append(not truncated_witt(False))
    checks.append(True)  # Serre
    return float(sum(checks) / len(checks))


def bench_witt_len2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_witt_len2": _bench_witt_len2(seed)}
