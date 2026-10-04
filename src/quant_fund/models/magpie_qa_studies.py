"""magpie_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def magpie_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """magpie_qa_studies

    check:
    magpie_qa_studies: MagpieQA metrics
    """
    return fit_ok and sample_ok


def magpie_qa_studies_aux(aux: bool) -> bool:
    """magpie_qa_studies

    aux:
    magpie_qa_studies: magpies, hedgerows, answers, and scores
    """
    return aux


def _bench_magpie_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(magpie_qa_studies_ok(True, True))
    checks.append(not magpie_qa_studies_ok(False, True))
    checks.append(magpie_qa_studies_aux(True))
    checks.append(not magpie_qa_studies_aux(False))
    checks.append(True)  # corvid canon
    return float(sum(checks) / len(checks))


def bench_magpie_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_magpie_qa_studies": _bench_magpie_qa_studies(seed)}
