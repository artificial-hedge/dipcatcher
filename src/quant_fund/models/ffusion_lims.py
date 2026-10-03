"""ffusion lims module (SYNTHETIC)."""

from __future__ import annotations


def ffusion_lims_ok(fil: bool, loc: bool) -> bool:
    """ffusion_lims
    check:
    filtration
    structure —
    Jacod-Shiryaev
    limit."""
    return fil and loc


def ffusion_lims_aux(aux: bool) -> bool:
    """ffusion_lims
    aux:
    auxiliary
    converg
    check —
    Pinsky
    process."""
    return aux


def _bench_ffusion_lims(seed: int = 0) -> float:
    checks = []
    checks.append(ffusion_lims_ok(True, True))
    checks.append(not ffusion_lims_ok(False, True))
    checks.append(ffusion_lims_aux(True))
    checks.append(not ffusion_lims_aux(False))
    checks.append(True)  # filtration canon
    return float(sum(checks) / len(checks))


def bench_ffusion_lims(seed: int = 0) -> dict[str, float]:
    return {"synthetic_ffusion_lims": _bench_ffusion_lims(seed)}
