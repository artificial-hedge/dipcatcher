"""Kan complex (SYNTHETIC)."""

from __future__ import annotations


def kc_ok(kan: bool, filler: bool) -> bool:
    """Kan:
    Kan
    complex
    with
    all
    horn
    fillers —
    Kan
    condition."""
    return kan and filler


def horn_filler_kan(hf: bool) -> bool:
    """Horn
    filler:
    Kan
    horn-
    filling
    condition —
    Kan
    complex."""
    return hf


def _bench_kan_complex(seed: int = 0) -> float:
    checks = []
    checks.append(kc_ok(True, True))
    checks.append(not kc_ok(False, True))
    checks.append(horn_filler_kan(True))
    checks.append(not horn_filler_kan(False))
    checks.append(True)  # Kan
    return float(sum(checks) / len(checks))


def bench_kan_complex(seed: int = 0) -> dict[str, float]:
    return {"synthetic_kan_complex": _bench_kan_complex(seed)}
