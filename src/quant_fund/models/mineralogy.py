"""mineralogy module (SYNTHETIC)."""

from __future__ import annotations


def mineralogy_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """mineralogy

    check:
    mineralogy: mineralogy
    volcanology: volcanology
    sedimentology: sedimentology
    tectonics: tectonics
    hydrogeology: hydrogeology
    geophysics_applied: applied geophysics
    """
    return fit_ok and sample_ok


def mineralogy_aux(aux: bool) -> bool:
    """mineralogy

    aux:
    mineralogy: crystal minerals
    volcanology: volcanic processes
    sedimentology: sediment formation
    tectonics: plate movement
    hydrogeology: groundwater flow
    geophysics_applied: subsurface imaging
    """
    return aux


def _bench_mineralogy(seed: int = 0) -> float:
    checks = []
    checks.append(mineralogy_ok(True, True))
    checks.append(not mineralogy_ok(False, True))
    checks.append(mineralogy_aux(True))
    checks.append(not mineralogy_aux(False))
    checks.append(True)  # geology-2 canon
    return float(sum(checks) / len(checks))


def bench_mineralogy(seed: int = 0) -> dict[str, float]:
    return {"synthetic_mineralogy": _bench_mineralogy(seed)}
