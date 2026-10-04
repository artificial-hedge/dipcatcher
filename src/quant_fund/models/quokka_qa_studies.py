"""quokka_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def quokka_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """quokka_qa_studies

    check:
    quokka_qa_studies: QuokkaQA metrics
    """
    return fit_ok and sample_ok


def quokka_qa_studies_aux(aux: bool) -> bool:
    """quokka_qa_studies

    aux:
    quokka_qa_studies: quokkas, islets, answers, and scores
    """
    return aux


def _bench_quokka_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(quokka_qa_studies_ok(True, True))
    checks.append(not quokka_qa_studies_ok(False, True))
    checks.append(quokka_qa_studies_aux(True))
    checks.append(not quokka_qa_studies_aux(False))
    checks.append(True)  # marsupial canon
    return float(sum(checks) / len(checks))


def bench_quokka_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_quokka_qa_studies": _bench_quokka_qa_studies(seed)}
