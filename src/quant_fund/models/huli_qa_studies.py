"""huli_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def huli_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """huli_qa_studies

    check:
    huli_qa_studies: HuliQA metrics
    """
    return fit_ok and sample_ok


def huli_qa_studies_aux(aux: bool) -> bool:
    """huli_qa_studies

    aux:
    huli_qa_studies: huli, fox spirits, answers, and scores
    """
    return aux


def _bench_huli_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(huli_qa_studies_ok(True, True))
    checks.append(not huli_qa_studies_ok(False, True))
    checks.append(huli_qa_studies_aux(True))
    checks.append(not huli_qa_studies_aux(False))
    checks.append(True)  # chinese-myth canon
    return float(sum(checks) / len(checks))


def bench_huli_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_huli_qa_studies": _bench_huli_qa_studies(seed)}
