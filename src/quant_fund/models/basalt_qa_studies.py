"""basalt_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def basalt_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """basalt_qa_studies

    check:
    basalt_qa_studies: BasaltQA metrics
    """
    return fit_ok and sample_ok


def basalt_qa_studies_aux(aux: bool) -> bool:
    """basalt_qa_studies

    aux:
    basalt_qa_studies: basalts, columns, answers, and scores
    """
    return aux


def _bench_basalt_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(basalt_qa_studies_ok(True, True))
    checks.append(not basalt_qa_studies_ok(False, True))
    checks.append(basalt_qa_studies_aux(True))
    checks.append(not basalt_qa_studies_aux(False))
    checks.append(True)  # bedrock canon
    return float(sum(checks) / len(checks))


def bench_basalt_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_basalt_qa_studies": _bench_basalt_qa_studies(seed)}
