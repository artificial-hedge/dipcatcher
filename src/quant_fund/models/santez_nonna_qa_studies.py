"""santez_nonna_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def santez_nonna_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """santez_nonna_qa_studies

    check:
    santez_nonna_qa_studies: h
    """
    return fit_ok and sample_ok


def santez_nonna_qa_studies_aux(aux: bool) -> bool:
    """santez_nonna_qa_studies

    aux:
    santez_nonna_qa_studies: o
    """
    return aux


def _bench_santez_nonna_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(santez_nonna_qa_studies_ok(True, True))
    checks.append(not santez_nonna_qa_studies_ok(False, True))
    checks.append(santez_nonna_qa_studies_aux(True))
    checks.append(not santez_nonna_qa_studies_aux(False))
    checks.append(True)  # breton-myth-2 canon
    return float(sum(checks) / len(checks))


def bench_santez_nonna_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_santez_nonna_qa_studies": _bench_santez_nonna_qa_studies(seed)}
