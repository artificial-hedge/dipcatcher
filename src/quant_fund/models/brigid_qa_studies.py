"""brigid_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def brigid_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """brigid_qa_studies

    check:
    brigid_qa_studies: BrigidQA metrics
    """
    return fit_ok and sample_ok


def brigid_qa_studies_aux(aux: bool) -> bool:
    """brigid_qa_studies

    aux:
    brigid_qa_studies: brigid, hearth flames, answers, and scores
    """
    return aux


def _bench_brigid_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(brigid_qa_studies_ok(True, True))
    checks.append(not brigid_qa_studies_ok(False, True))
    checks.append(brigid_qa_studies_aux(True))
    checks.append(not brigid_qa_studies_aux(False))
    checks.append(True)  # irish-myth canon
    return float(sum(checks) / len(checks))


def bench_brigid_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_brigid_qa_studies": _bench_brigid_qa_studies(seed)}
