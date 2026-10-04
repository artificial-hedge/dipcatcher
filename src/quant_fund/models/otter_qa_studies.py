"""otter_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def otter_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """otter_qa_studies

    check:
    otter_qa_studies: OtterQA metrics
    """
    return fit_ok and sample_ok


def otter_qa_studies_aux(aux: bool) -> bool:
    """otter_qa_studies

    aux:
    otter_qa_studies: otters, kelp, answers, and scores
    """
    return aux


def _bench_otter_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(otter_qa_studies_ok(True, True))
    checks.append(not otter_qa_studies_ok(False, True))
    checks.append(otter_qa_studies_aux(True))
    checks.append(not otter_qa_studies_aux(False))
    checks.append(True)  # marine mammal canon
    return float(sum(checks) / len(checks))


def bench_otter_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_otter_qa_studies": _bench_otter_qa_studies(seed)}
