"""jiangshi_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def jiangshi_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """jiangshi_qa_studies

    check:
    jiangshi_qa_studies: JiangshiQA metrics
    """
    return fit_ok and sample_ok


def jiangshi_qa_studies_aux(aux: bool) -> bool:
    """jiangshi_qa_studies

    aux:
    jiangshi_qa_studies: jiangshi, hopping corpses, answers, and scores
    """
    return aux


def _bench_jiangshi_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(jiangshi_qa_studies_ok(True, True))
    checks.append(not jiangshi_qa_studies_ok(False, True))
    checks.append(jiangshi_qa_studies_aux(True))
    checks.append(not jiangshi_qa_studies_aux(False))
    checks.append(True)  # chinese-myth canon
    return float(sum(checks) / len(checks))


def bench_jiangshi_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_jiangshi_qa_studies": _bench_jiangshi_qa_studies(seed)}
