"""oryx_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def oryx_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """oryx_qa_studies

    check:
    oryx_qa_studies: OryxQA metrics
    """
    return fit_ok and sample_ok


def oryx_qa_studies_aux(aux: bool) -> bool:
    """oryx_qa_studies

    aux:
    oryx_qa_studies: oryxes, horns, answers, and scores
    """
    return aux


def _bench_oryx_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(oryx_qa_studies_ok(True, True))
    checks.append(not oryx_qa_studies_ok(False, True))
    checks.append(oryx_qa_studies_aux(True))
    checks.append(not oryx_qa_studies_aux(False))
    checks.append(True)  # antelope canon
    return float(sum(checks) / len(checks))


def bench_oryx_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_oryx_qa_studies": _bench_oryx_qa_studies(seed)}
