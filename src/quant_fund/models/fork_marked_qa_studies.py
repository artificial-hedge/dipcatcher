"""fork_marked_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def fork_marked_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """fork_marked_qa_studies

    check:
    fork_marked_qa_studies: ForkMarkedQA metrics
    """
    return fit_ok and sample_ok


def fork_marked_qa_studies_aux(aux: bool) -> bool:
    """fork_marked_qa_studies

    aux:
    fork_marked_qa_studies: fork-marked lemurs, dry tree crowns, answers, and scores
    """
    return aux


def _bench_fork_marked_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(fork_marked_qa_studies_ok(True, True))
    checks.append(not fork_marked_qa_studies_ok(False, True))
    checks.append(fork_marked_qa_studies_aux(True))
    checks.append(not fork_marked_qa_studies_aux(False))
    checks.append(True)  # lemur-3 canon
    return float(sum(checks) / len(checks))


def bench_fork_marked_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_fork_marked_qa_studies": _bench_fork_marked_qa_studies(seed)}
