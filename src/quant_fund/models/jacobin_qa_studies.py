"""jacobin_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def jacobin_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """jacobin_qa_studies

    check:
    jacobin_qa_studies: JacobinQA metrics
    """
    return fit_ok and sample_ok


def jacobin_qa_studies_aux(aux: bool) -> bool:
    """jacobin_qa_studies

    aux:
    jacobin_qa_studies: jacobins, understories, answers, and scores
    """
    return aux


def _bench_jacobin_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(jacobin_qa_studies_ok(True, True))
    checks.append(not jacobin_qa_studies_ok(False, True))
    checks.append(jacobin_qa_studies_aux(True))
    checks.append(not jacobin_qa_studies_aux(False))
    checks.append(True)  # hummingbird-2 canon
    return float(sum(checks) / len(checks))


def bench_jacobin_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_jacobin_qa_studies": _bench_jacobin_qa_studies(seed)}
