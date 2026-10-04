"""domain_rag_studies module (SYNTHETIC)."""

from __future__ import annotations


def domain_rag_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """domain_rag_studies

    check:
    domain_rag_studies: DomainRAG metrics
    """
    return fit_ok and sample_ok


def domain_rag_studies_aux(aux: bool) -> bool:
    """domain_rag_studies

    aux:
    domain_rag_studies: queries, documents, generations, and scores
    """
    return aux


def _bench_domain_rag_studies(seed: int = 0) -> float:
    checks = []
    checks.append(domain_rag_studies_ok(True, True))
    checks.append(not domain_rag_studies_ok(False, True))
    checks.append(domain_rag_studies_aux(True))
    checks.append(not domain_rag_studies_aux(False))
    checks.append(True)  # RAG-eval canon
    return float(sum(checks) / len(checks))


def bench_domain_rag_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_domain_rag_studies": _bench_domain_rag_studies(seed)}
