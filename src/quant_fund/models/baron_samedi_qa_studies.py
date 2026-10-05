"""baron_samedi_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def baron_samedi_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """baron_samedi_qa_studies

    check:
    baron_samedi_qa_studies: B
    """
    return fit_ok and sample_ok


def baron_samedi_qa_studies_aux(aux: bool) -> bool:
    """baron_samedi_qa_studies

    aux:
    baron_samedi_qa_studies: a
    """
    return aux


def _bench_baron_samedi_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(baron_samedi_qa_studies_ok(True, True))
    checks.append(not baron_samedi_qa_studies_ok(False, True))
    checks.append(baron_samedi_qa_studies_aux(True))
    checks.append(not baron_samedi_qa_studies_aux(False))
    checks.append(True)  # vodou-loa canon
    return float(sum(checks) / len(checks))


def bench_baron_samedi_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_baron_samedi_qa_studies": _bench_baron_samedi_qa_studies(seed)}
