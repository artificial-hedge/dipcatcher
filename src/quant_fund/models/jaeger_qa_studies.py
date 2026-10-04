"""jaeger_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def jaeger_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """jaeger_qa_studies

    check:
    jaeger_qa_studies: JaegerQA metrics
    """
    return fit_ok and sample_ok


def jaeger_qa_studies_aux(aux: bool) -> bool:
    """jaeger_qa_studies

    aux:
    jaeger_qa_studies: jaegers, skuas, answers, and scores
    """
    return aux


def _bench_jaeger_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(jaeger_qa_studies_ok(True, True))
    checks.append(not jaeger_qa_studies_ok(False, True))
    checks.append(jaeger_qa_studies_aux(True))
    checks.append(not jaeger_qa_studies_aux(False))
    checks.append(True)  # seabird-2 canon
    return float(sum(checks) / len(checks))


def bench_jaeger_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_jaeger_qa_studies": _bench_jaeger_qa_studies(seed)}
