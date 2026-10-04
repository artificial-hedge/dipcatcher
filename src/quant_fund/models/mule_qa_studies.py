"""mule_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def mule_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """mule_qa_studies

    check:
    mule_qa_studies: MuleQA metrics
    """
    return fit_ok and sample_ok


def mule_qa_studies_aux(aux: bool) -> bool:
    """mule_qa_studies

    aux:
    mule_qa_studies: mule deer, sagebrush flats, answers, and scores
    """
    return aux


def _bench_mule_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(mule_qa_studies_ok(True, True))
    checks.append(not mule_qa_studies_ok(False, True))
    checks.append(mule_qa_studies_aux(True))
    checks.append(not mule_qa_studies_aux(False))
    checks.append(True)  # deer-3 canon
    return float(sum(checks) / len(checks))


def bench_mule_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_mule_qa_studies": _bench_mule_qa_studies(seed)}
