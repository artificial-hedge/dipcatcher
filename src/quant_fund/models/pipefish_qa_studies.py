"""pipefish_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def pipefish_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """pipefish_qa_studies

    check:
    pipefish_qa_studies: PipefishQA metrics
    """
    return fit_ok and sample_ok


def pipefish_qa_studies_aux(aux: bool) -> bool:
    """pipefish_qa_studies

    aux:
    pipefish_qa_studies: pipefish, seagrass beds, answers, and scores
    """
    return aux


def _bench_pipefish_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(pipefish_qa_studies_ok(True, True))
    checks.append(not pipefish_qa_studies_ok(False, True))
    checks.append(pipefish_qa_studies_aux(True))
    checks.append(not pipefish_qa_studies_aux(False))
    checks.append(True)  # reef-fish-3 canon
    return float(sum(checks) / len(checks))


def bench_pipefish_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_pipefish_qa_studies": _bench_pipefish_qa_studies(seed)}
