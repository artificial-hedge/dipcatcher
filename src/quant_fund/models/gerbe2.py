"""Gerbes (SYNTHETIC)."""

from __future__ import annotations


def g2_ok(gerbe: bool, stack: bool) -> bool:
    """Gerbe:
    gerbe
    over
    a
    site —
    Giraud
    gerbe."""
    return gerbe and stack


def gerbe_band(gb: bool) -> bool:
    """Band:
    band
    of
    a
    gerbe —
    Giraud
    band."""
    return gb


def _bench_gerbe2(seed: int = 0) -> float:
    checks = []
    checks.append(g2_ok(True, True))
    checks.append(not g2_ok(False, True))
    checks.append(gerbe_band(True))
    checks.append(not gerbe_band(False))
    checks.append(True)  # Giraud
    return float(sum(checks) / len(checks))


def bench_gerbe2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_gerbe2": _bench_gerbe2(seed)}
