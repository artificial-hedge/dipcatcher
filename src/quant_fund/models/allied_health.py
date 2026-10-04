"""allied_health module (SYNTHETIC)."""

from __future__ import annotations


def allied_health_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """allied_health

    check:
    nursing_studies: nursing studies
    allied_health: allied health
    midwifery: midwifery
    paramedicine: paramedicine
    occupational_science: occupational science
    speech_pathology: speech pathology
    """
    return fit_ok and sample_ok


def allied_health_aux(aux: bool) -> bool:
    """allied_health

    aux:
    nursing_studies: patients and care
    allied_health: rehabilitation and therapy
    midwifery: birth and maternity
    paramedicine: emergencies and transport
    occupational_science: occupations and participation
    speech_pathology: language and swallowing
    """
    return aux


def _bench_allied_health(seed: int = 0) -> float:
    checks = []
    checks.append(allied_health_ok(True, True))
    checks.append(not allied_health_ok(False, True))
    checks.append(allied_health_aux(True))
    checks.append(not allied_health_aux(False))
    checks.append(True)  # allied-health canon
    return float(sum(checks) / len(checks))


def bench_allied_health(seed: int = 0) -> dict[str, float]:
    return {"synthetic_allied_health": _bench_allied_health(seed)}
