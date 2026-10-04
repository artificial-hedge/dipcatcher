"""uakari_2_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def uakari_2_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """uakari_2_qa_studies

    check:
    uakari_2_qa_studies: UakariTwoQA metrics
    """
    return fit_ok and sample_ok


def uakari_2_qa_studies_aux(aux: bool) -> bool:
    """uakari_2_qa_studies

    aux:
    uakari_2_qa_studies: uakaris, varzea forests, answers, and scores
    """
    return aux


def _bench_uakari_2_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(uakari_2_qa_studies_ok(True, True))
    checks.append(not uakari_2_qa_studies_ok(False, True))
    checks.append(uakari_2_qa_studies_aux(True))
    checks.append(not uakari_2_qa_studies_aux(False))
    checks.append(True)  # primate-4 canon
    return float(sum(checks) / len(checks))


def bench_uakari_2_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_uakari_2_qa_studies": _bench_uakari_2_qa_studies(seed)}
