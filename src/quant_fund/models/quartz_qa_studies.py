"""quartz_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def quartz_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """quartz_qa_studies

    check:
    quartz_qa_studies: QuartzQA metrics
    """
    return fit_ok and sample_ok


def quartz_qa_studies_aux(aux: bool) -> bool:
    """quartz_qa_studies

    aux:
    quartz_qa_studies: quartzes, veins, answers, and scores
    """
    return aux


def _bench_quartz_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(quartz_qa_studies_ok(True, True))
    checks.append(not quartz_qa_studies_ok(False, True))
    checks.append(quartz_qa_studies_aux(True))
    checks.append(not quartz_qa_studies_aux(False))
    checks.append(True)  # mineral canon
    return float(sum(checks) / len(checks))


def bench_quartz_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_quartz_qa_studies": _bench_quartz_qa_studies(seed)}
