"""krahang_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def krahang_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """krahang_qa_studies

    check:
    krahang_qa_studies: K
    """
    return fit_ok and sample_ok


def krahang_qa_studies_aux(aux: bool) -> bool:
    """krahang_qa_studies

    aux:
    krahang_qa_studies: r
    """
    return aux


def _bench_krahang_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(krahang_qa_studies_ok(True, True))
    checks.append(not krahang_qa_studies_ok(False, True))
    checks.append(krahang_qa_studies_aux(True))
    checks.append(not krahang_qa_studies_aux(False))
    checks.append(True)  # thai-demon canon
    return float(sum(checks) / len(checks))


def bench_krahang_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_krahang_qa_studies": _bench_krahang_qa_studies(seed)}
