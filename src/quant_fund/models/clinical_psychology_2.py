"""clinical_psychology_2 module (SYNTHETIC)."""

from __future__ import annotations


def clinical_psychology_2_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """clinical_psychology_2

    check:
    midwifery_studies: midwifery studies
    orthoptics: orthoptics
    audiology_studies: audiology studies
    opticianry: opticianry
    prosthetics_orthotics: prosthetics orthotics
    clinical_psychology_2: clinical psychology 2
    """
    return fit_ok and sample_ok


def clinical_psychology_2_aux(aux: bool) -> bool:
    """clinical_psychology_2

    aux:
    midwifery_studies: birth and prenatal
    orthoptics: binocular vision
    audiology_studies: hearing and balance
    opticianry: lenses and fittings
    prosthetics_orthotics: limbs and braces
    clinical_psychology_2: assessment and therapy
    """
    return aux


def _bench_clinical_psychology_2(seed: int = 0) -> float:
    checks = []
    checks.append(clinical_psychology_2_ok(True, True))
    checks.append(not clinical_psychology_2_ok(False, True))
    checks.append(clinical_psychology_2_aux(True))
    checks.append(not clinical_psychology_2_aux(False))
    checks.append(True)  # allied-health-2 canon
    return float(sum(checks) / len(checks))


def bench_clinical_psychology_2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_clinical_psychology_2": _bench_clinical_psychology_2(seed)}
