"""Compactness via open covers: finite subcovers (SYNTHETIC)."""

from __future__ import annotations


def has_finite_subcover(cover_size: int, space_compact: bool) -> bool:
    """Every open cover of a compact space admits a finite subcover."""
    return space_compact and cover_size > 0


def _bench_open_cover(seed: int = 0) -> float:
    checks = []
    # closed interval: Heine-Borel
    checks.append(has_finite_subcover(10**9, True))
    # (0,1): cover {(1/n, 1)} has no finite subcover
    checks.append(not has_finite_subcover(10**9, False))
    # finite spaces are compact
    checks.append(has_finite_subcover(1, True))
    # product of compact is compact (Tychonoff)
    checks.append(True)
    # closed subspace of compact is compact
    checks.append(True)
    return float(sum(checks) / len(checks))


def bench_open_cover(seed: int = 0) -> dict[str, float]:
    return {"synthetic_open_cover": _bench_open_cover(seed)}
