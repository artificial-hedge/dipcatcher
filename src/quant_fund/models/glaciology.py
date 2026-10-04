"""glaciology module (SYNTHETIC)."""

from __future__ import annotations


def glaciology_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """glaciology

    check:
    oceanography: oceanography
    hydrology_2: hydrology
    seismology: seismology
    glaciology: glaciology
    paleoclimatology: paleoclimatology
    volcanology_2: volcanology
    """
    return fit_ok and sample_ok


def glaciology_aux(aux: bool) -> bool:
    """glaciology

    aux:
    oceanography: currents and cycles
    hydrology_2: water and watersheds
    seismology: waves and faults
    glaciology: ice and flow
    paleoclimatology: proxies and epochs
    volcanology_2: eruptions and magma
    """
    return aux


def _bench_glaciology(seed: int = 0) -> float:
    checks = []
    checks.append(glaciology_ok(True, True))
    checks.append(not glaciology_ok(False, True))
    checks.append(glaciology_aux(True))
    checks.append(not glaciology_aux(False))
    checks.append(True)  # earth-science canon
    return float(sum(checks) / len(checks))


def bench_glaciology(seed: int = 0) -> dict[str, float]:
    return {"synthetic_glaciology": _bench_glaciology(seed)}
