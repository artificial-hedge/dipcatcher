"""manta_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def manta_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """manta_qa_studies

    check:
    manta_qa_studies: MantaQA metrics
    """
    return fit_ok and sample_ok


def manta_qa_studies_aux(aux: bool) -> bool:
    """manta_qa_studies

    aux:
    manta_qa_studies: manta rays, open ocean, answers, and scores
    """
    return aux


def _bench_manta_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(manta_qa_studies_ok(True, True))
    checks.append(not manta_qa_studies_ok(False, True))
    checks.append(manta_qa_studies_aux(True))
    checks.append(not manta_qa_studies_aux(False))
    checks.append(True)  # ray canon
    return float(sum(checks) / len(checks))


def bench_manta_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_manta_qa_studies": _bench_manta_qa_studies(seed)}
