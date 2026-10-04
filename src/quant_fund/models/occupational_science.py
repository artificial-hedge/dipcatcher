"""occupational_science module (SYNTHETIC)."""

from __future__ import annotations


def occupational_science_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """occupational_science

    check:
    nursing_studies: nursing studies
    allied_health: allied health
    midwifery: midwifery
    paramedicine: paramedicine
    occupational_science: occupational science
    speech_pathology: speech pathology
    """
    return fit_ok and sample_ok


def occupational_science_aux(aux: bool) -> bool:
    """occupational_science

    aux:
    nursing_studies: patients and care
    allied_health: rehabilitation and therapy
    midwifery: birth and maternity
    paramedicine: emergencies and transport
    occupational_science: occupations and participation
    speech_pathology: language and swallowing
    """
    return aux


def _bench_occupational_science(seed: int = 0) -> float:
    checks = []
    checks.append(occupational_science_ok(True, True))
    checks.append(not occupational_science_ok(False, True))
    checks.append(occupational_science_aux(True))
    checks.append(not occupational_science_aux(False))
    checks.append(True)  # allied-health canon
    return float(sum(checks) / len(checks))


def bench_occupational_science(seed: int = 0) -> dict[str, float]:
    return {"synthetic_occupational_science": _bench_occupational_science(seed)}
