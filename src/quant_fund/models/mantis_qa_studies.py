"""mantis_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def mantis_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """mantis_qa_studies

    check:
    mantis_qa_studies: MantisQA metrics
    """
    return fit_ok and sample_ok


def mantis_qa_studies_aux(aux: bool) -> bool:
    """mantis_qa_studies

    aux:
    mantis_qa_studies: mantises, strikes, answers, and scores
    """
    return aux


def _bench_mantis_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(mantis_qa_studies_ok(True, True))
    checks.append(not mantis_qa_studies_ok(False, True))
    checks.append(mantis_qa_studies_aux(True))
    checks.append(not mantis_qa_studies_aux(False))
    checks.append(True)  # invertebrate canon
    return float(sum(checks) / len(checks))


def bench_mantis_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_mantis_qa_studies": _bench_mantis_qa_studies(seed)}
