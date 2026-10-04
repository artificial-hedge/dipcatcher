"""bilby_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def bilby_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """bilby_qa_studies

    check:
    bilby_qa_studies: BilbyQA metrics
    """
    return fit_ok and sample_ok


def bilby_qa_studies_aux(aux: bool) -> bool:
    """bilby_qa_studies

    aux:
    bilby_qa_studies: bilbies, deserts, answers, and scores
    """
    return aux


def _bench_bilby_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(bilby_qa_studies_ok(True, True))
    checks.append(not bilby_qa_studies_ok(False, True))
    checks.append(bilby_qa_studies_aux(True))
    checks.append(not bilby_qa_studies_aux(False))
    checks.append(True)  # marsupial-2 canon
    return float(sum(checks) / len(checks))


def bench_bilby_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_bilby_qa_studies": _bench_bilby_qa_studies(seed)}
