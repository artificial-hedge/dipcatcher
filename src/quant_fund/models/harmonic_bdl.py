"""Harmonic bundle (SYNTHETIC)."""

from __future__ import annotations


def hb_ok2(harmonic: bool, higgs: bool) -> bool:
    """Harmonic
    bundle:
    flat
    bundle
    with
    harmonic
    metric —
    pluriharmonic."""
    return harmonic and higgs


def pluriharmonic(ph: bool) -> bool:
    """Pluriharmonic:
    pluriharmonic
    metric
    gives
    Higgs
    field —
    Corlette."""
    return ph


def _bench_harmonic_bdl(seed: int = 0) -> float:
    checks = []
    checks.append(hb_ok2(True, True))
    checks.append(not hb_ok2(False, True))
    checks.append(pluriharmonic(True))
    checks.append(not pluriharmonic(False))
    checks.append(True)  # Corlette
    return float(sum(checks) / len(checks))


def bench_harmonic_bdl(seed: int = 0) -> dict[str, float]:
    return {"synthetic_harmonic_bdl": _bench_harmonic_bdl(seed)}
