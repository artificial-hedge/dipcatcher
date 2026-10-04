"""ensem_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def ensem_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """ensem_qa_studies

    check:
    ensem_qa_studies: EnsemQA metrics
    """
    return fit_ok and sample_ok


def ensem_qa_studies_aux(aux: bool) -> bool:
    """ensem_qa_studies

    aux:
    ensem_qa_studies: documents, hops, answers, and scores
    """
    return aux


def _bench_ensem_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(ensem_qa_studies_ok(True, True))
    checks.append(not ensem_qa_studies_ok(False, True))
    checks.append(ensem_qa_studies_aux(True))
    checks.append(not ensem_qa_studies_aux(False))
    checks.append(True)  # multi-hop-QA-2 canon
    return float(sum(checks) / len(checks))


def bench_ensem_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_ensem_qa_studies": _bench_ensem_qa_studies(seed)}
