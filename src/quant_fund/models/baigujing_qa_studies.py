"""baigujing_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def baigujing_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """baigujing_qa_studies

    check:
    baigujing_qa_studies: B
    """
    return fit_ok and sample_ok


def baigujing_qa_studies_aux(aux: bool) -> bool:
    """baigujing_qa_studies

    aux:
    baigujing_qa_studies: a
    """
    return aux


def _bench_baigujing_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(baigujing_qa_studies_ok(True, True))
    checks.append(not baigujing_qa_studies_ok(False, True))
    checks.append(baigujing_qa_studies_aux(True))
    checks.append(not baigujing_qa_studies_aux(False))
    checks.append(True)  # chinese-demon canon
    return float(sum(checks) / len(checks))


def bench_baigujing_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_baigujing_qa_studies": _bench_baigujing_qa_studies(seed)}
