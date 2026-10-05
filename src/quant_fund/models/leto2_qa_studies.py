"""leto2_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def leto2_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """leto2_qa_studies

    check:
    leto2_qa_studies: Leto2QA metrics
    """
    return fit_ok and sample_ok


def leto2_qa_studies_aux(aux: bool) -> bool:
    """leto2_qa_studies

    aux:
    leto2_qa_studies: leto2, twin mothers, answers, and scores
    """
    return aux


def _bench_leto2_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(leto2_qa_studies_ok(True, True))
    checks.append(not leto2_qa_studies_ok(False, True))
    checks.append(leto2_qa_studies_aux(True))
    checks.append(not leto2_qa_studies_aux(False))
    checks.append(True)  # lycian-myth canon
    return float(sum(checks) / len(checks))


def bench_leto2_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_leto2_qa_studies": _bench_leto2_qa_studies(seed)}
