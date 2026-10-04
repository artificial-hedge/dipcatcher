"""darkman_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def darkman_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """darkman_qa_studies

    check:
    darkman_qa_studies: d
    """
    return fit_ok and sample_ok


def darkman_qa_studies_aux(aux: bool) -> bool:
    """darkman_qa_studies

    aux:
    darkman_qa_studies: o
    """
    return aux


def _bench_darkman_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(darkman_qa_studies_ok(True, True))
    checks.append(not darkman_qa_studies_ok(False, True))
    checks.append(darkman_qa_studies_aux(True))
    checks.append(not darkman_qa_studies_aux(False))
    checks.append(True)  # breton-myth-2 canon
    return float(sum(checks) / len(checks))


def bench_darkman_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_darkman_qa_studies": _bench_darkman_qa_studies(seed)}
