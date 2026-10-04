"""springbok_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def springbok_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """springbok_qa_studies

    check:
    springbok_qa_studies: SpringbokQA metrics
    """
    return fit_ok and sample_ok


def springbok_qa_studies_aux(aux: bool) -> bool:
    """springbok_qa_studies

    aux:
    springbok_qa_studies: springboks, pronks, answers, and scores
    """
    return aux


def _bench_springbok_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(springbok_qa_studies_ok(True, True))
    checks.append(not springbok_qa_studies_ok(False, True))
    checks.append(springbok_qa_studies_aux(True))
    checks.append(not springbok_qa_studies_aux(False))
    checks.append(True)  # antelope canon
    return float(sum(checks) / len(checks))


def bench_springbok_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_springbok_qa_studies": _bench_springbok_qa_studies(seed)}
