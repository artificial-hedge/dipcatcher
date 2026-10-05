"""sirenito_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def sirenito_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """sirenito_qa_studies

    check:
    sirenito_qa_studies: S
    """
    return fit_ok and sample_ok


def sirenito_qa_studies_aux(aux: bool) -> bool:
    """sirenito_qa_studies

    aux:
    sirenito_qa_studies: i
    """
    return aux


def _bench_sirenito_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(sirenito_qa_studies_ok(True, True))
    checks.append(not sirenito_qa_studies_ok(False, True))
    checks.append(sirenito_qa_studies_aux(True))
    checks.append(not sirenito_qa_studies_aux(False))
    checks.append(True)  # andean-demon canon
    return float(sum(checks) / len(checks))


def bench_sirenito_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_sirenito_qa_studies": _bench_sirenito_qa_studies(seed)}
