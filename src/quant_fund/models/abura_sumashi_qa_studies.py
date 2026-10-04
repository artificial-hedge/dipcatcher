"""abura_sumashi_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def abura_sumashi_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """abura_sumashi_qa_studies

    check:
    abura_sumashi_qa_studies: AburaSumashiQA metrics
    """
    return fit_ok and sample_ok


def abura_sumashi_qa_studies_aux(aux: bool) -> bool:
    """abura_sumashi_qa_studies

    aux:
    abura_sumashi_qa_studies: abura-sumashis, mountain passes, answers, and scores
    """
    return aux


def _bench_abura_sumashi_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(abura_sumashi_qa_studies_ok(True, True))
    checks.append(not abura_sumashi_qa_studies_ok(False, True))
    checks.append(abura_sumashi_qa_studies_aux(True))
    checks.append(not abura_sumashi_qa_studies_aux(False))
    checks.append(True)  # yokai-3 canon
    return float(sum(checks) / len(checks))


def bench_abura_sumashi_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_abura_sumashi_qa_studies": _bench_abura_sumashi_qa_studies(seed)}
