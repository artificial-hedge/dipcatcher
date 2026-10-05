"""jiuying_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def jiuying_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """jiuying_qa_studies

    check:
    jiuying_qa_studies: J
    """
    return fit_ok and sample_ok


def jiuying_qa_studies_aux(aux: bool) -> bool:
    """jiuying_qa_studies

    aux:
    jiuying_qa_studies: i
    """
    return aux


def _bench_jiuying_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(jiuying_qa_studies_ok(True, True))
    checks.append(not jiuying_qa_studies_ok(False, True))
    checks.append(jiuying_qa_studies_aux(True))
    checks.append(not jiuying_qa_studies_aux(False))
    checks.append(True)  # chinese-demon canon
    return float(sum(checks) / len(checks))


def bench_jiuying_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_jiuying_qa_studies": _bench_jiuying_qa_studies(seed)}
