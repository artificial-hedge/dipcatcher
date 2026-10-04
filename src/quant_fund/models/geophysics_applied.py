"""geophysics_applied module (SYNTHETIC)."""

from __future__ import annotations


def geophysics_applied_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """geophysics_applied

    check:
    mineralogy: mineralogy
    volcanology: volcanology
    sedimentology: sedimentology
    tectonics: tectonics
    hydrogeology: hydrogeology
    geophysics_applied: applied geophysics
    """
    return fit_ok and sample_ok


def geophysics_applied_aux(aux: bool) -> bool:
    """geophysics_applied

    aux:
    mineralogy: crystal minerals
    volcanology: volcanic processes
    sedimentology: sediment formation
    tectonics: plate movement
    hydrogeology: groundwater flow
    geophysics_applied: subsurface imaging
    """
    return aux


def _bench_geophysics_applied(seed: int = 0) -> float:
    checks = []
    checks.append(geophysics_applied_ok(True, True))
    checks.append(not geophysics_applied_ok(False, True))
    checks.append(geophysics_applied_aux(True))
    checks.append(not geophysics_applied_aux(False))
    checks.append(True)  # geology-2 canon
    return float(sum(checks) / len(checks))


def bench_geophysics_applied(seed: int = 0) -> dict[str, float]:
    return {"synthetic_geophysics_applied": _bench_geophysics_applied(seed)}
