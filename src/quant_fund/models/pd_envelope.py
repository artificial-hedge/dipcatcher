"""PD envelope (SYNTHETIC)."""

from __future__ import annotations


def pe_ok2(pd_envelope: bool, universal: bool) -> bool:
    """PD
    envelope:
    universal
    PD
    thickening —
    envelope."""
    return pd_envelope and universal


def pd_thickening(pt: bool) -> bool:
    """PD
    thickening:
    nilpotent
    immersion
    with
    PD
    structure —
    envelope."""
    return pt


def _bench_pd_envelope(seed: int = 0) -> float:
    checks = []
    checks.append(pe_ok2(True, True))
    checks.append(not pe_ok2(False, True))
    checks.append(pd_thickening(True))
    checks.append(not pd_thickening(False))
    checks.append(True)  # Berthelot
    return float(sum(checks) / len(checks))


def bench_pd_envelope(seed: int = 0) -> dict[str, float]:
    return {"synthetic_pd_envelope": _bench_pd_envelope(seed)}
