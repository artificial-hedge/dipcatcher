"""agga_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def agga_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """agga_qa_studies

    check:
    agga_qa_studies: AggaQA metrics
    """
    return fit_ok and sample_ok


def agga_qa_studies_aux(aux: bool) -> bool:
    """agga_qa_studies

    aux:
    agga_qa_studies: agga, siege walls, answers, and scores
    """
    return aux


def _bench_agga_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(agga_qa_studies_ok(True, True))
    checks.append(not agga_qa_studies_ok(False, True))
    checks.append(agga_qa_studies_aux(True))
    checks.append(not agga_qa_studies_aux(False))
    checks.append(True)  # sumerian-5 canon
    return float(sum(checks) / len(checks))


def bench_agga_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_agga_qa_studies": _bench_agga_qa_studies(seed)}
