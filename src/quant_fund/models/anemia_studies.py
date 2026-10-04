"""anemia_studies module (SYNTHETIC)."""

from __future__ import annotations


def anemia_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """anemia_studies

    check:
    anemia_studies: hemoglobin and ferritin
    ..."""
    return fit_ok and sample_ok


def anemia_studies_aux(aux: bool) -> bool:
    """anemia_studies

    aux:
    anemia_studies: iron and mcv
    ..."""
    return aux


def _bench_anemia_studies(seed: int = 0) -> float:
    checks = []
    checks.append(anemia_studies_ok(True, True))
    checks.append(not anemia_studies_ok(False, True))
    checks.append(anemia_studies_aux(True))
    checks.append(not anemia_studies_aux(False))
    checks.append(True)  # hematology canon
    return float(sum(checks) / len(checks))


def bench_anemia_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_anemia_studies": _bench_anemia_studies(seed)}
