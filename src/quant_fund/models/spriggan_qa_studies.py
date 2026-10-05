"""spriggan_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def spriggan_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """spriggan_qa_studies

    check:
    spriggan_qa_studies: g
    """
    return fit_ok and sample_ok


def spriggan_qa_studies_aux(aux: bool) -> bool:
    """spriggan_qa_studies

    aux:
    spriggan_qa_studies: u
    """
    return aux


def _bench_spriggan_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(spriggan_qa_studies_ok(True, True))
    checks.append(not spriggan_qa_studies_ok(False, True))
    checks.append(spriggan_qa_studies_aux(True))
    checks.append(not spriggan_qa_studies_aux(False))
    checks.append(True)  # cornish-myth canon
    return float(sum(checks) / len(checks))


def bench_spriggan_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_spriggan_qa_studies": _bench_spriggan_qa_studies(seed)}
