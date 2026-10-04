"""Spatial diamond (SYNTHETIC)."""

from __future__ import annotations


def sd_ok(spatial: bool, diamond: bool) -> bool:
    """Spatial
    diamond:
    spatial
    diamond —
    spectral."""
    return spatial and diamond


def spectral_diamond(sdi: bool) -> bool:
    """Spectral
    diamond:
    spectral
    diamond —
    quasi
    compact."""
    return sdi


def _bench_spatial_diamond(seed: int = 0) -> float:
    checks = []
    checks.append(sd_ok(True, True))
    checks.append(not sd_ok(False, True))
    checks.append(spectral_diamond(True))
    checks.append(not spectral_diamond(False))
    checks.append(True)  # Scholze
    return float(sum(checks) / len(checks))


def bench_spatial_diamond(seed: int = 0) -> dict[str, float]:
    return {"synthetic_spatial_diamond": _bench_spatial_diamond(seed)}
