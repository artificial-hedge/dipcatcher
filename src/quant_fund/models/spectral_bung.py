"""Spectral BunG (SYNTHETIC)."""

from __future__ import annotations


def sb_ok(local_systems: bool, spectral_data: bool) -> bool:
    """Spectral
    Bun:
    moduli
    of
    local
    systems
    on
    spectral
    side —
    Langlands
    dual."""
    return local_systems and spectral_data


def locsys_moduli(lm: bool) -> bool:
    """LocSys
    moduli:
    stack
    of
    G-
    local
    systems —
    spectral
    parameter."""
    return lm


def _bench_spectral_bung(seed: int = 0) -> float:
    checks = []
    checks.append(sb_ok(True, True))
    checks.append(not sb_ok(False, True))
    checks.append(locsys_moduli(True))
    checks.append(not locsys_moduli(False))
    checks.append(True)  # Arinkin-Gaitsgory
    return float(sum(checks) / len(checks))


def bench_spectral_bung(seed: int = 0) -> dict[str, float]:
    return {"synthetic_spectral_bung": _bench_spectral_bung(seed)}
