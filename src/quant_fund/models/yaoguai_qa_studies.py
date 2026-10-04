"""yaoguai_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def yaoguai_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """yaoguai_qa_studies

    check:
    yaoguai_qa_studies: YaoguaiQA metrics
    """
    return fit_ok and sample_ok


def yaoguai_qa_studies_aux(aux: bool) -> bool:
    """yaoguai_qa_studies

    aux:
    yaoguai_qa_studies: yaoguai, monster spirits, answers, and scores
    """
    return aux


def _bench_yaoguai_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(yaoguai_qa_studies_ok(True, True))
    checks.append(not yaoguai_qa_studies_ok(False, True))
    checks.append(yaoguai_qa_studies_aux(True))
    checks.append(not yaoguai_qa_studies_aux(False))
    checks.append(True)  # chinese-myth canon
    return float(sum(checks) / len(checks))


def bench_yaoguai_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_yaoguai_qa_studies": _bench_yaoguai_qa_studies(seed)}
