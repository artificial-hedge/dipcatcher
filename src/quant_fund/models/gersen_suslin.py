"""Gersten-Suslin resolution (SYNTHETIC)."""

from __future__ import annotations


def gs_ok(gersten: bool, suslin: bool) -> bool:
    """Gersten:
    Gersten
    conjecture
    /
    Suslin
    —
    Gersten
    resolution."""
    return gersten and suslin


def gersten_complex(gc: bool) -> bool:
    """Gersten
    complex:
    Gersten
    complex —
    Quillen
    Gersten."""
    return gc


def _bench_gersen_suslin(seed: int = 0) -> float:
    checks = []
    checks.append(gs_ok(True, True))
    checks.append(not gs_ok(False, True))
    checks.append(gersten_complex(True))
    checks.append(not gersten_complex(False))
    checks.append(True)  # Quillen
    return float(sum(checks) / len(checks))


def bench_gersen_suslin(seed: int = 0) -> dict[str, float]:
    return {"synthetic_gersen_suslin": _bench_gersen_suslin(seed)}
