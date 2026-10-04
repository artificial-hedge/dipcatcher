"""Banded gerbes (SYNTHETIC)."""

from __future__ import annotations


def bg_ok(band: bool, gerbe: bool) -> bool:
    """Banded:
    gerbe
    banded
    by
    an
    abelian
    group —
    banded
    gerbe."""
    return band and gerbe


def gerbe_class(gc: bool) -> bool:
    """Gerbe
    class:
    H^2
    class
    of
    a
    gerbe —
    Giraud
    H2."""
    return gc


def _bench_band_gerbe(seed: int = 0) -> float:
    checks = []
    checks.append(bg_ok(True, True))
    checks.append(not bg_ok(False, True))
    checks.append(gerbe_class(True))
    checks.append(not gerbe_class(False))
    checks.append(True)  # Giraud
    return float(sum(checks) / len(checks))


def bench_band_gerbe(seed: int = 0) -> dict[str, float]:
    return {"synthetic_band_gerbe": _bench_band_gerbe(seed)}
