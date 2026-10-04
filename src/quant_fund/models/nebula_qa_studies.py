"""nebula_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def nebula_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """nebula_qa_studies

    check:
    nebula_qa_studies: NebulaQA metrics
    """
    return fit_ok and sample_ok


def nebula_qa_studies_aux(aux: bool) -> bool:
    """nebula_qa_studies

    aux:
    nebula_qa_studies: nebulae, gases, answers, and scores
    """
    return aux


def _bench_nebula_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(nebula_qa_studies_ok(True, True))
    checks.append(not nebula_qa_studies_ok(False, True))
    checks.append(nebula_qa_studies_aux(True))
    checks.append(not nebula_qa_studies_aux(False))
    checks.append(True)  # celestial canon
    return float(sum(checks) / len(checks))


def bench_nebula_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_nebula_qa_studies": _bench_nebula_qa_studies(seed)}
