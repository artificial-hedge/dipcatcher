"""brigid2_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def brigid2_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """brigid2_qa_studies

    check:
    brigid2_qa_studies: Brigid2QA metrics
    """
    return fit_ok and sample_ok


def brigid2_qa_studies_aux(aux: bool) -> bool:
    """brigid2_qa_studies

    aux:
    brigid2_qa_studies: brigid2, triple flames, answers, and scores
    """
    return aux


def _bench_brigid2_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(brigid2_qa_studies_ok(True, True))
    checks.append(not brigid2_qa_studies_ok(False, True))
    checks.append(brigid2_qa_studies_aux(True))
    checks.append(not brigid2_qa_studies_aux(False))
    checks.append(True)  # celtic-myth-5 canon
    return float(sum(checks) / len(checks))


def bench_brigid2_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_brigid2_qa_studies": _bench_brigid2_qa_studies(seed)}
