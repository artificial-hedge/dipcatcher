"""dongwanggong_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def dongwanggong_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """dongwanggong_qa_studies

    check:
    dongwanggong_qa_studies: DongwanggongQA metrics
    """
    return fit_ok and sample_ok


def dongwanggong_qa_studies_aux(aux: bool) -> bool:
    """dongwanggong_qa_studies

    aux:
    dongwanggong_qa_studies: dongwanggong, eastern king fathers, answers, and scores
    """
    return aux


def _bench_dongwanggong_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(dongwanggong_qa_studies_ok(True, True))
    checks.append(not dongwanggong_qa_studies_ok(False, True))
    checks.append(dongwanggong_qa_studies_aux(True))
    checks.append(not dongwanggong_qa_studies_aux(False))
    checks.append(True)  # chinese-myth-2 canon
    return float(sum(checks) / len(checks))


def bench_dongwanggong_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_dongwanggong_qa_studies": _bench_dongwanggong_qa_studies(seed)}
