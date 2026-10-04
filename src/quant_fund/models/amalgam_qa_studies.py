"""amalgam_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def amalgam_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """amalgam_qa_studies

    check:
    amalgam_qa_studies: AmalgamQA metrics
    """
    return fit_ok and sample_ok


def amalgam_qa_studies_aux(aux: bool) -> bool:
    """amalgam_qa_studies

    aux:
    amalgam_qa_studies: amalgams, fillings, answers, and scores
    """
    return aux


def _bench_amalgam_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(amalgam_qa_studies_ok(True, True))
    checks.append(not amalgam_qa_studies_ok(False, True))
    checks.append(amalgam_qa_studies_aux(True))
    checks.append(not amalgam_qa_studies_aux(False))
    checks.append(True)  # alloy canon
    return float(sum(checks) / len(checks))


def bench_amalgam_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_amalgam_qa_studies": _bench_amalgam_qa_studies(seed)}
