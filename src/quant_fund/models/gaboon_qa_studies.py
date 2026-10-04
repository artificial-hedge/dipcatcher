"""gaboon_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def gaboon_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """gaboon_qa_studies

    check:
    gaboon_qa_studies: GaboonQA metrics
    """
    return fit_ok and sample_ok


def gaboon_qa_studies_aux(aux: bool) -> bool:
    """gaboon_qa_studies

    aux:
    gaboon_qa_studies: gaboon vipers, rainforest floors, answers, and scores
    """
    return aux


def _bench_gaboon_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(gaboon_qa_studies_ok(True, True))
    checks.append(not gaboon_qa_studies_ok(False, True))
    checks.append(gaboon_qa_studies_aux(True))
    checks.append(not gaboon_qa_studies_aux(False))
    checks.append(True)  # venom-2 canon
    return float(sum(checks) / len(checks))


def bench_gaboon_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_gaboon_qa_studies": _bench_gaboon_qa_studies(seed)}
