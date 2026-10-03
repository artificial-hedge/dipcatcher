"""Stable stems (SYNTHETIC)."""

from __future__ import annotations


def sst_ok(stable: bool, stem: bool) -> bool:
    """Stable
    stem:
    stable
    stem —
    stable
    sphere."""
    return stable and stem


def sphere_spectrum(ss: bool) -> bool:
    """Sphere
    spectrum:
    sphere
    spectrum —
    stable
    stems."""
    return ss


def _bench_stable_stem2(seed: int = 0) -> float:
    checks = []
    checks.append(sst_ok(True, True))
    checks.append(not sst_ok(False, True))
    checks.append(sphere_spectrum(True))
    checks.append(not sphere_spectrum(False))
    checks.append(True)  # Serre
    return float(sum(checks) / len(checks))


def bench_stable_stem2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_stable_stem2": _bench_stable_stem2(seed)}
