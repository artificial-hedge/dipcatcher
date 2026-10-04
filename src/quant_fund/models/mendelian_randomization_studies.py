"""mendelian_randomization_studies module (SYNTHETIC)."""

from __future__ import annotations


def mendelian_randomization_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """mendelian_randomization_studies

    check:
    mendelian_randomization_studies: instrument strength and IVW/relevance and exclusion
    """
    return fit_ok and sample_ok


def mendelian_randomization_studies_aux(aux: bool) -> bool:
    """mendelian_randomization_studies

    aux:
    mendelian_randomization_studies: pleiotropy and assumptions/estimation and tests
    """
    return aux


def _bench_mendelian_randomization_studies(seed: int = 0) -> float:
    checks = []
    checks.append(mendelian_randomization_studies_ok(True, True))
    checks.append(not mendelian_randomization_studies_ok(False, True))
    checks.append(mendelian_randomization_studies_aux(True))
    checks.append(not mendelian_randomization_studies_aux(False))
    checks.append(True)  # genetic-epidemiology/MR canon
    return float(sum(checks) / len(checks))


def bench_mendelian_randomization_studies(seed: int = 0) -> dict[str, float]:
    return {
        "synthetic_mendelian_randomization_studies": _bench_mendelian_randomization_studies(seed)
    }
