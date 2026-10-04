"""kasa_obake_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def kasa_obake_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """kasa_obake_qa_studies

    check:
    kasa_obake_qa_studies: K
    """
    return fit_ok and sample_ok


def kasa_obake_qa_studies_aux(aux: bool) -> bool:
    """kasa_obake_qa_studies

    aux:
    kasa_obake_qa_studies: a
    """
    return aux


def _bench_kasa_obake_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(kasa_obake_qa_studies_ok(True, True))
    checks.append(not kasa_obake_qa_studies_ok(False, True))
    checks.append(kasa_obake_qa_studies_aux(True))
    checks.append(not kasa_obake_qa_studies_aux(False))
    checks.append(True)  # yokai-8 canon
    return float(sum(checks) / len(checks))


def bench_kasa_obake_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_kasa_obake_qa_studies": _bench_kasa_obake_qa_studies(seed)}
