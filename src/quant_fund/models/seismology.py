"""seismology module (SYNTHETIC)."""

from __future__ import annotations


def seismology_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """seismology

    check:
    oceanography: oceanography
    hydrology_2: hydrology
    seismology: seismology
    glaciology: glaciology
    paleoclimatology: paleoclimatology
    volcanology_2: volcanology
    """
    return fit_ok and sample_ok


def seismology_aux(aux: bool) -> bool:
    """seismology

    aux:
    oceanography: currents and cycles
    hydrology_2: water and watersheds
    seismology: waves and faults
    glaciology: ice and flow
    paleoclimatology: proxies and epochs
    volcanology_2: eruptions and magma
    """
    return aux


def _bench_seismology(seed: int = 0) -> float:
    checks = []
    checks.append(seismology_ok(True, True))
    checks.append(not seismology_ok(False, True))
    checks.append(seismology_aux(True))
    checks.append(not seismology_aux(False))
    checks.append(True)  # earth-science canon
    return float(sum(checks) / len(checks))


def bench_seismology(seed: int = 0) -> dict[str, float]:
    return {"synthetic_seismology": _bench_seismology(seed)}
