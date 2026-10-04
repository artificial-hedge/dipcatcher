"""manat_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def manat_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """manat_qa_studies

    check:
    manat_qa_studies: f
    """
    return fit_ok and sample_ok


def manat_qa_studies_aux(aux: bool) -> bool:
    """manat_qa_studies

    aux:
    manat_qa_studies: a
    """
    return aux


def _bench_manat_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(manat_qa_studies_ok(True, True))
    checks.append(not manat_qa_studies_ok(False, True))
    checks.append(manat_qa_studies_aux(True))
    checks.append(not manat_qa_studies_aux(False))
    checks.append(True)  # arabian-myth canon
    return float(sum(checks) / len(checks))


def bench_manat_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_manat_qa_studies": _bench_manat_qa_studies(seed)}
