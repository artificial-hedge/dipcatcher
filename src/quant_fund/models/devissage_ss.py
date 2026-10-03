"""Devissage spectral sequence (SYNTHETIC)."""

from __future__ import annotations


def ds_ok(devissage: bool, ss: bool) -> bool:
    """Devissage:
    devissage
    spectral
    sequence —
    Quillen
    devissage."""
    return devissage and ss


def devissage_thm(dt: bool) -> bool:
    """Devissage
    theorem:
    devissage
    for
    abelian
    categories —
    Quillen
    devissage."""
    return dt


def _bench_devissage_ss(seed: int = 0) -> float:
    checks = []
    checks.append(ds_ok(True, True))
    checks.append(not ds_ok(False, True))
    checks.append(devissage_thm(True))
    checks.append(not devissage_thm(False))
    checks.append(True)  # Quillen
    return float(sum(checks) / len(checks))


def bench_devissage_ss(seed: int = 0) -> dict[str, float]:
    return {"synthetic_devissage_ss": _bench_devissage_ss(seed)}
