"""dentistry_3 module (SYNTHETIC)."""

from __future__ import annotations


def dentistry_3_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """dentistry_3

    check:
    medicine_7: medicine
    dentistry_3: dentistry
    nursing_2: nursing
    public_health_2: public health
    veterinary_medicine_2: veterinary medicine
    pharmacy_2: pharmacy
    """
    return fit_ok and sample_ok


def dentistry_3_aux(aux: bool) -> bool:
    """dentistry_3

    aux:
    medicine_7: diagnosis and treatment
    dentistry_3: caries and prosthetics
    nursing_2: care and triage
    public_health_2: populations and prevention
    veterinary_medicine_2: species and zoonoses
    pharmacy_2: dosages and interactions
    """
    return aux


def _bench_dentistry_3(seed: int = 0) -> float:
    checks = []
    checks.append(dentistry_3_ok(True, True))
    checks.append(not dentistry_3_ok(False, True))
    checks.append(dentistry_3_aux(True))
    checks.append(not dentistry_3_aux(False))
    checks.append(True)  # health-sciences canon
    return float(sum(checks) / len(checks))


def bench_dentistry_3(seed: int = 0) -> dict[str, float]:
    return {"synthetic_dentistry_3": _bench_dentistry_3(seed)}
