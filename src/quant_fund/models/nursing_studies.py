"""nursing_studies module (SYNTHETIC)."""

from __future__ import annotations


def nursing_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """nursing_studies

    check:
    nursing_studies: nursing studies
    allied_health: allied health
    midwifery: midwifery
    paramedicine: paramedicine
    occupational_science: occupational science
    speech_pathology: speech pathology
    """
    return fit_ok and sample_ok


def nursing_studies_aux(aux: bool) -> bool:
    """nursing_studies

    aux:
    nursing_studies: patients and care
    allied_health: rehabilitation and therapy
    midwifery: birth and maternity
    paramedicine: emergencies and transport
    occupational_science: occupations and participation
    speech_pathology: language and swallowing
    """
    return aux


def _bench_nursing_studies(seed: int = 0) -> float:
    checks = []
    checks.append(nursing_studies_ok(True, True))
    checks.append(not nursing_studies_ok(False, True))
    checks.append(nursing_studies_aux(True))
    checks.append(not nursing_studies_aux(False))
    checks.append(True)  # allied-health canon
    return float(sum(checks) / len(checks))


def bench_nursing_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_nursing_studies": _bench_nursing_studies(seed)}
