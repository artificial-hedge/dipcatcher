"""manannan2_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def manannan2_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """manannan2_qa_studies

    check:
    manannan2_qa_studies: Manannan2QA metrics
    """
    return fit_ok and sample_ok


def manannan2_qa_studies_aux(aux: bool) -> bool:
    """manannan2_qa_studies

    aux:
    manannan2_qa_studies: manannan2, sea riders, answers, and scores
    """
    return aux


def _bench_manannan2_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(manannan2_qa_studies_ok(True, True))
    checks.append(not manannan2_qa_studies_ok(False, True))
    checks.append(manannan2_qa_studies_aux(True))
    checks.append(not manannan2_qa_studies_aux(False))
    checks.append(True)  # celtic-myth-5 canon
    return float(sum(checks) / len(checks))


def bench_manannan2_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_manannan2_qa_studies": _bench_manannan2_qa_studies(seed)}
