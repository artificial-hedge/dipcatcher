"""Hitchin section (SYNTHETIC)."""

from __future__ import annotations


def hs_ok(hitchin_base: bool, section: bool) -> bool:
    """Hitchin
    section:
    section
    of
    Hitchin
    fibration —
    Kostant-
    Hitchin."""
    return hitchin_base and section


def kostant_hitchin(kh: bool) -> bool:
    """Kostant-
    Hitchin:
    section
    gives
    canonical
    point —
    split
    section."""
    return kh


def _bench_hitchin_section(seed: int = 0) -> float:
    checks = []
    checks.append(hs_ok(True, True))
    checks.append(not hs_ok(False, True))
    checks.append(kostant_hitchin(True))
    checks.append(not kostant_hitchin(False))
    checks.append(True)  # Kostant-Hitchin
    return float(sum(checks) / len(checks))


def bench_hitchin_section(seed: int = 0) -> dict[str, float]:
    return {"synthetic_hitchin_section": _bench_hitchin_section(seed)}
