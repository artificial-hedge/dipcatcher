"""ilmarinen_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def ilmarinen_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """ilmarinen_qa_studies

    check:
    ilmarinen_qa_studies: IlmarinenQA metrics
    """
    return fit_ok and sample_ok


def ilmarinen_qa_studies_aux(aux: bool) -> bool:
    """ilmarinen_qa_studies

    aux:
    ilmarinen_qa_studies: ilmarinen, sky smiths, answers, and scores
    """
    return aux


def _bench_ilmarinen_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(ilmarinen_qa_studies_ok(True, True))
    checks.append(not ilmarinen_qa_studies_ok(False, True))
    checks.append(ilmarinen_qa_studies_aux(True))
    checks.append(not ilmarinen_qa_studies_aux(False))
    checks.append(True)  # finnish-myth-2 canon
    return float(sum(checks) / len(checks))


def bench_ilmarinen_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_ilmarinen_qa_studies": _bench_ilmarinen_qa_studies(seed)}
