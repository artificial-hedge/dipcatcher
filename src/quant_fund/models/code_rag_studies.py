"""code_rag_studies module (SYNTHETIC)."""

from __future__ import annotations


def code_rag_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """code_rag_studies

    check:
    code_rag_studies: Code retrieval-augmented generation metrics
    """
    return fit_ok and sample_ok


def code_rag_studies_aux(aux: bool) -> bool:
    """code_rag_studies

    aux:
    code_rag_studies: repositories, queries, retrievals, and accuracies
    """
    return aux


def _bench_code_rag_studies(seed: int = 0) -> float:
    checks = []
    checks.append(code_rag_studies_ok(True, True))
    checks.append(not code_rag_studies_ok(False, True))
    checks.append(code_rag_studies_aux(True))
    checks.append(not code_rag_studies_aux(False))
    checks.append(True)  # code-eval-4 canon
    return float(sum(checks) / len(checks))


def bench_code_rag_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_code_rag_studies": _bench_code_rag_studies(seed)}
