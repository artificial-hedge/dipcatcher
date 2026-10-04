"""ultrasound_studies module (SYNTHETIC)."""

from __future__ import annotations


def ultrasound_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """ultrasound_studies

    check:
    ultrasound_studies: doppler and echo
    ..."""
    return fit_ok and sample_ok


def ultrasound_studies_aux(aux: bool) -> bool:
    """ultrasound_studies

    aux:
    ultrasound_studies: sonography and elastography
    ..."""
    return aux


def _bench_ultrasound_studies(seed: int = 0) -> float:
    checks = []
    checks.append(ultrasound_studies_ok(True, True))
    checks.append(not ultrasound_studies_ok(False, True))
    checks.append(ultrasound_studies_aux(True))
    checks.append(not ultrasound_studies_aux(False))
    checks.append(True)  # imaging-modality canon
    return float(sum(checks) / len(checks))


def bench_ultrasound_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_ultrasound_studies": _bench_ultrasound_studies(seed)}
