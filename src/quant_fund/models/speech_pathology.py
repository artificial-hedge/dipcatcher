"""speech_pathology module (SYNTHETIC)."""

from __future__ import annotations


def speech_pathology_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """speech_pathology

    check:
    nursing_studies: nursing studies
    allied_health: allied health
    midwifery: midwifery
    paramedicine: paramedicine
    occupational_science: occupational science
    speech_pathology: speech pathology
    """
    return fit_ok and sample_ok


def speech_pathology_aux(aux: bool) -> bool:
    """speech_pathology

    aux:
    nursing_studies: patients and care
    allied_health: rehabilitation and therapy
    midwifery: birth and maternity
    paramedicine: emergencies and transport
    occupational_science: occupations and participation
    speech_pathology: language and swallowing
    """
    return aux


def _bench_speech_pathology(seed: int = 0) -> float:
    checks = []
    checks.append(speech_pathology_ok(True, True))
    checks.append(not speech_pathology_ok(False, True))
    checks.append(speech_pathology_aux(True))
    checks.append(not speech_pathology_aux(False))
    checks.append(True)  # allied-health canon
    return float(sum(checks) / len(checks))


def bench_speech_pathology(seed: int = 0) -> dict[str, float]:
    return {"synthetic_speech_pathology": _bench_speech_pathology(seed)}
