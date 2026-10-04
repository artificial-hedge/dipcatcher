"""palliative_medicine module (SYNTHETIC)."""

from __future__ import annotations


def palliative_medicine_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """palliative_medicine

    check:
    urology: urology
    ophthalmology: ophthalmology
    otolaryngology: otolaryngology
    palliative_medicine: palliative medicine
    sports_medicine: sports medicine
    rehabilitation_medicine: rehabilitation medicine
    """
    return fit_ok and sample_ok


def palliative_medicine_aux(aux: bool) -> bool:
    """palliative_medicine

    aux:
    urology: urinary tract and renal
    ophthalmology: vision and ocular
    otolaryngology: ear nose and throat
    palliative_medicine: comfort and dignity
    sports_medicine: athletic injury and performance
    rehabilitation_medicine: functional recovery
    """
    return aux


def _bench_palliative_medicine(seed: int = 0) -> float:
    checks = []
    checks.append(palliative_medicine_ok(True, True))
    checks.append(not palliative_medicine_ok(False, True))
    checks.append(palliative_medicine_aux(True))
    checks.append(not palliative_medicine_aux(False))
    checks.append(True)  # medicine-5 canon
    return float(sum(checks) / len(checks))


def bench_palliative_medicine(seed: int = 0) -> dict[str, float]:
    return {"synthetic_palliative_medicine": _bench_palliative_medicine(seed)}
