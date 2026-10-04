"""hydrogeology module (SYNTHETIC)."""

from __future__ import annotations


def hydrogeology_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """hydrogeology

    check:
    mineralogy: mineralogy
    volcanology: volcanology
    sedimentology: sedimentology
    tectonics: tectonics
    hydrogeology: hydrogeology
    geophysics_applied: applied geophysics
    """
    return fit_ok and sample_ok


def hydrogeology_aux(aux: bool) -> bool:
    """hydrogeology

    aux:
    mineralogy: crystal minerals
    volcanology: volcanic processes
    sedimentology: sediment formation
    tectonics: plate movement
    hydrogeology: groundwater flow
    geophysics_applied: subsurface imaging
    """
    return aux


def _bench_hydrogeology(seed: int = 0) -> float:
    checks = []
    checks.append(hydrogeology_ok(True, True))
    checks.append(not hydrogeology_ok(False, True))
    checks.append(hydrogeology_aux(True))
    checks.append(not hydrogeology_aux(False))
    checks.append(True)  # geology-2 canon
    return float(sum(checks) / len(checks))


def bench_hydrogeology(seed: int = 0) -> dict[str, float]:
    return {"synthetic_hydrogeology": _bench_hydrogeology(seed)}
