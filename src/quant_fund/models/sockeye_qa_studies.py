"""sockeye_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def sockeye_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """sockeye_qa_studies

    check:
    sockeye_qa_studies: SockeyeQA metrics
    """
    return fit_ok and sample_ok


def sockeye_qa_studies_aux(aux: bool) -> bool:
    """sockeye_qa_studies

    aux:
    sockeye_qa_studies: sockeye salmon, spawning grounds, answers, and scores
    """
    return aux


def _bench_sockeye_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(sockeye_qa_studies_ok(True, True))
    checks.append(not sockeye_qa_studies_ok(False, True))
    checks.append(sockeye_qa_studies_aux(True))
    checks.append(not sockeye_qa_studies_aux(False))
    checks.append(True)  # salmonid canon
    return float(sum(checks) / len(checks))


def bench_sockeye_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_sockeye_qa_studies": _bench_sockeye_qa_studies(seed)}
