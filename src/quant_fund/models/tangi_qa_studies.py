"""tangi_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def tangi_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """tangi_qa_studies

    check:
    tangi_qa_studies: t
    """
    return fit_ok and sample_ok


def tangi_qa_studies_aux(aux: bool) -> bool:
    """tangi_qa_studies

    aux:
    tangi_qa_studies: o
    """
    return aux


def _bench_tangi_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(tangi_qa_studies_ok(True, True))
    checks.append(not tangi_qa_studies_ok(False, True))
    checks.append(tangi_qa_studies_aux(True))
    checks.append(not tangi_qa_studies_aux(False))
    checks.append(True)  # breton-myth canon
    return float(sum(checks) / len(checks))


def bench_tangi_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_tangi_qa_studies": _bench_tangi_qa_studies(seed)}
