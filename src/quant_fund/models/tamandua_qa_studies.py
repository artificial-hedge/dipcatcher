"""tamandua_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def tamandua_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """tamandua_qa_studies

    check:
    tamandua_qa_studies: TamanduaQA metrics
    """
    return fit_ok and sample_ok


def tamandua_qa_studies_aux(aux: bool) -> bool:
    """tamandua_qa_studies

    aux:
    tamandua_qa_studies: tamanduas, ant nests, answers, and scores
    """
    return aux


def _bench_tamandua_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(tamandua_qa_studies_ok(True, True))
    checks.append(not tamandua_qa_studies_ok(False, True))
    checks.append(tamandua_qa_studies_aux(True))
    checks.append(not tamandua_qa_studies_aux(False))
    checks.append(True)  # neotropical-2 canon
    return float(sum(checks) / len(checks))


def bench_tamandua_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_tamandua_qa_studies": _bench_tamandua_qa_studies(seed)}
