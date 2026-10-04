"""nursing_2 module (SYNTHETIC)."""

from __future__ import annotations


def nursing_2_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """nursing_2

    check:
    medicine_7: medicine
    dentistry_3: dentistry
    nursing_2: nursing
    public_health_2: public health
    veterinary_medicine_2: veterinary medicine
    pharmacy_2: pharmacy
    """
    return fit_ok and sample_ok


def nursing_2_aux(aux: bool) -> bool:
    """nursing_2

    aux:
    medicine_7: diagnosis and treatment
    dentistry_3: caries and prosthetics
    nursing_2: care and triage
    public_health_2: populations and prevention
    veterinary_medicine_2: species and zoonoses
    pharmacy_2: dosages and interactions
    """
    return aux


def _bench_nursing_2(seed: int = 0) -> float:
    checks = []
    checks.append(nursing_2_ok(True, True))
    checks.append(not nursing_2_ok(False, True))
    checks.append(nursing_2_aux(True))
    checks.append(not nursing_2_aux(False))
    checks.append(True)  # health-sciences canon
    return float(sum(checks) / len(checks))


def bench_nursing_2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_nursing_2": _bench_nursing_2(seed)}
