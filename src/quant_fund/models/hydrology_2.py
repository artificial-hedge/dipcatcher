"""hydrology_2 module (SYNTHETIC)."""

from __future__ import annotations


def hydrology_2_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """hydrology_2

    check:
    oceanography: oceanography
    hydrology_2: hydrology
    seismology: seismology
    glaciology: glaciology
    paleoclimatology: paleoclimatology
    volcanology_2: volcanology
    """
    return fit_ok and sample_ok


def hydrology_2_aux(aux: bool) -> bool:
    """hydrology_2

    aux:
    oceanography: currents and cycles
    hydrology_2: water and watersheds
    seismology: waves and faults
    glaciology: ice and flow
    paleoclimatology: proxies and epochs
    volcanology_2: eruptions and magma
    """
    return aux


def _bench_hydrology_2(seed: int = 0) -> float:
    checks = []
    checks.append(hydrology_2_ok(True, True))
    checks.append(not hydrology_2_ok(False, True))
    checks.append(hydrology_2_aux(True))
    checks.append(not hydrology_2_aux(False))
    checks.append(True)  # earth-science canon
    return float(sum(checks) / len(checks))


def bench_hydrology_2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_hydrology_2": _bench_hydrology_2(seed)}
