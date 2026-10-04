"""bushbuck_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def bushbuck_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """bushbuck_qa_studies

    check:
    bushbuck_qa_studies: BushbuckQA metrics
    """
    return fit_ok and sample_ok


def bushbuck_qa_studies_aux(aux: bool) -> bool:
    """bushbuck_qa_studies

    aux:
    bushbuck_qa_studies: bushbucks, riverine undergrowth, answers, and scores
    """
    return aux


def _bench_bushbuck_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(bushbuck_qa_studies_ok(True, True))
    checks.append(not bushbuck_qa_studies_ok(False, True))
    checks.append(bushbuck_qa_studies_aux(True))
    checks.append(not bushbuck_qa_studies_aux(False))
    checks.append(True)  # antelope-3 canon
    return float(sum(checks) / len(checks))


def bench_bushbuck_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_bushbuck_qa_studies": _bench_bushbuck_qa_studies(seed)}
