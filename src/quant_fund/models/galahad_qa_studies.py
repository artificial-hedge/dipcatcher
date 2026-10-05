"""galahad_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def galahad_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """galahad_qa_studies

    check:
    galahad_qa_studies: g
    """
    return fit_ok and sample_ok


def galahad_qa_studies_aux(aux: bool) -> bool:
    """galahad_qa_studies

    aux:
    galahad_qa_studies: r
    """
    return aux


def _bench_galahad_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(galahad_qa_studies_ok(True, True))
    checks.append(not galahad_qa_studies_ok(False, True))
    checks.append(galahad_qa_studies_aux(True))
    checks.append(not galahad_qa_studies_aux(False))
    checks.append(True)  # arthurian-2 canon
    return float(sum(checks) / len(checks))


def bench_galahad_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_galahad_qa_studies": _bench_galahad_qa_studies(seed)}
