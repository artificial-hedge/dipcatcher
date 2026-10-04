"""flow_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def flow_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """flow_qa_studies

    check:
    flow_qa_studies: FlowQA metrics
    """
    return fit_ok and sample_ok


def flow_qa_studies_aux(aux: bool) -> bool:
    """flow_qa_studies

    aux:
    flow_qa_studies: flows, steps, answers, and scores
    """
    return aux


def _bench_flow_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(flow_qa_studies_ok(True, True))
    checks.append(not flow_qa_studies_ok(False, True))
    checks.append(flow_qa_studies_aux(True))
    checks.append(not flow_qa_studies_aux(False))
    checks.append(True)  # instruction-task canon
    return float(sum(checks) / len(checks))


def bench_flow_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_flow_qa_studies": _bench_flow_qa_studies(seed)}
