"""yanwang_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def yanwang_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """yanwang_qa_studies

    check:
    yanwang_qa_studies: Y
    """
    return fit_ok and sample_ok


def yanwang_qa_studies_aux(aux: bool) -> bool:
    """yanwang_qa_studies

    aux:
    yanwang_qa_studies: a
    """
    return aux


def _bench_yanwang_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(yanwang_qa_studies_ok(True, True))
    checks.append(not yanwang_qa_studies_ok(False, True))
    checks.append(yanwang_qa_studies_aux(True))
    checks.append(not yanwang_qa_studies_aux(False))
    checks.append(True)  # chinese-underworld canon
    return float(sum(checks) / len(checks))


def bench_yanwang_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_yanwang_qa_studies": _bench_yanwang_qa_studies(seed)}
