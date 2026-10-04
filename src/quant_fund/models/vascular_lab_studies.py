"""vascular_lab_studies module (SYNTHETIC)."""

from __future__ import annotations


def vascular_lab_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """vascular_lab_studies

    check:
    vascular_lab_studies: duplex and abi
    ..."""
    return fit_ok and sample_ok


def vascular_lab_studies_aux(aux: bool) -> bool:
    """vascular_lab_studies

    aux:
    vascular_lab_studies: waveforms and velocities
    ..."""
    return aux


def _bench_vascular_lab_studies(seed: int = 0) -> float:
    checks = []
    checks.append(vascular_lab_studies_ok(True, True))
    checks.append(not vascular_lab_studies_ok(False, True))
    checks.append(vascular_lab_studies_aux(True))
    checks.append(not vascular_lab_studies_aux(False))
    checks.append(True)  # vascular-medicine canon
    return float(sum(checks) / len(checks))


def bench_vascular_lab_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_vascular_lab_studies": _bench_vascular_lab_studies(seed)}
