"""tamarin_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def tamarin_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """tamarin_qa_studies

    check:
    tamarin_qa_studies: TamarinQA metrics
    """
    return fit_ok and sample_ok


def tamarin_qa_studies_aux(aux: bool) -> bool:
    """tamarin_qa_studies

    aux:
    tamarin_qa_studies: tamarins, vine tangles, answers, and scores
    """
    return aux


def _bench_tamarin_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(tamarin_qa_studies_ok(True, True))
    checks.append(not tamarin_qa_studies_ok(False, True))
    checks.append(tamarin_qa_studies_aux(True))
    checks.append(not tamarin_qa_studies_aux(False))
    checks.append(True)  # primate canon
    return float(sum(checks) / len(checks))


def bench_tamarin_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_tamarin_qa_studies": _bench_tamarin_qa_studies(seed)}
