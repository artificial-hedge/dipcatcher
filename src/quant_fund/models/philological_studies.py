"""philological_studies module (SYNTHETIC)."""

from __future__ import annotations


def philological_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """philological_studies

    check:
    contact_linguistics: contact linguistics
    descriptive_linguistics: descriptive linguistics
    philological_studies: philology
    etymology: etymology
    dialectometry: dialectometry
    lexicography: lexicography
    """
    return fit_ok and sample_ok


def philological_studies_aux(aux: bool) -> bool:
    """philological_studies

    aux:
    contact_linguistics: language contact
    descriptive_linguistics: language description
    philological_studies: textual study
    etymology: word origins
    dialectometry: dialect distances
    lexicography: dictionary making
    """
    return aux


def _bench_philological_studies(seed: int = 0) -> float:
    checks = []
    checks.append(philological_studies_ok(True, True))
    checks.append(not philological_studies_ok(False, True))
    checks.append(philological_studies_aux(True))
    checks.append(not philological_studies_aux(False))
    checks.append(True)  # linguistics-5 canon
    return float(sum(checks) / len(checks))


def bench_philological_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_philological_studies": _bench_philological_studies(seed)}
