"""manx_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def manx_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """manx_qa_studies

    check:
    manx_qa_studies: ManxQA metrics
    """
    return fit_ok and sample_ok


def manx_qa_studies_aux(aux: bool) -> bool:
    """manx_qa_studies

    aux:
    manx_qa_studies: manx shearwaters, burrows, answers, and scores
    """
    return aux


def _bench_manx_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(manx_qa_studies_ok(True, True))
    checks.append(not manx_qa_studies_ok(False, True))
    checks.append(manx_qa_studies_aux(True))
    checks.append(not manx_qa_studies_aux(False))
    checks.append(True)  # pelagic canon
    return float(sum(checks) / len(checks))


def bench_manx_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_manx_qa_studies": _bench_manx_qa_studies(seed)}
